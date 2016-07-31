from flask import Flask, request, jsonify
from json import JSONEncoder
import json

import mysql.connector

import statistics

from datetime import timedelta
import datetime

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

@app.route('/setSample', methods=['POST'])
def setSample():
	result = "OK"
	models = json.loads(request.form['model'], object_hook = keystrokeDecoder);
	idUser = request.form['idUser'];
	insertKeystrokeDataInDB(idUser, models)
	return result;
	
@app.route('/validate', methods=['POST'])
def validate():
	models = json.loads(request.form['model'], object_hook = keystrokeDecoder);
	idUser = request.form['idUser'];
	insertKeystrokeDataInDB(idUser, models, True);
	# validator = ValidateKeys(selectOriginalModelsFromDB(idUser), selectFakeModelsFromDB(idUser));
	# validated = validator.validate(models);
	#return Encoder().encode(validated);
	return "OK";
	
def insertKeystrokeDataInDB(idUser, models, isTest = False):
	insertString = "INSERT INTO `mdl_user";
	insertKeystroke = "INSERT INTO mdl_user";
	if isTest == True:
		testePrefix = "_test";
		insertString = insertString + testePrefix; #
		insertKeystroke = insertKeystroke + testePrefix;
	insertString = (insertString + "_key_string`(`id_user`, `string`) VALUES (%s, %s)");
	insertKeystroke = (insertKeystroke + "_keystroke (key_code, time, action, id_string) "
				   "VALUES (%(key_code)s, %(time)s, %(action)s, %(id_string)s)");
	
	for model in models: # for each string
		string = createStringFromKeys(model); # for improve the performance I can update the string after the for bellow
		idString = executeSqlInDB(insertString, (idUser, string));
		insertNormalizedModelInDB(idUser, idString, model, isTest);
		for key in model:
			executeSqlInDB(insertKeystroke, getModelQuery(key, idString))

def createStringFromKeys(keys):
	string = "";
	for key in keys:
		string = string + '{}_'.format(key.keyCode);
	return string[: len(string) - 1];

def insertNormalizedModelInDB(idUser, idString, keystroke, isTest = False):
	insertNormalizedRecord = "INSERT INTO `mdl_user"
	updateNormalizedRecord = "UPDATE `mdl_user"
	
	if isTest == True:
		insertNormalizedRecord = insertNormalizedRecord + "_test";
		updateNormalizedRecord = updateNormalizedRecord +"_test";
	insertNormalizedRecord = insertNormalizedRecord + "_keystroke_normalized`(`id_user`, `id_string`) VALUES (%s, %s)";
	updateNormalizedRecord = updateNormalizedRecord + "_keystroke_normalized` ";
	
		
	executeSqlInDB(insertNormalizedRecord, (idUser, idString));
	
	keyDimensionsExtractor = KeystrokeDimensionsExtractor(keystroke);
	
	#extracting dimensions
	timePressed = keyDimensionsExtractor.getTimePressed();
	#geting avarage and standardDeviation
	timePressedAverage = statistics.mean(timePressed);
	timePressedstandardDeviation = statistics.pstdev(timePressed);
	
	latencies = keyDimensionsExtractor.getLatencies();
	latenciesAverage = statistics.mean(latencies);
	latenciesStandardDeviation = statistics.pstdev(latencies);
	
	dbModel = {
		'id_user': idUser,
		'id_string': idString,
		'press_average': timePressedAverage,
		'latency_avarage': latenciesAverage,
		'press_standard_deviation': timePressedstandardDeviation,
		'latency_standard_deviation': latenciesStandardDeviation,
	}
	
	#update in table created before
	updateNormalizedRecord = updateNormalizedRecord + (" SET `press_average`= %(press_average)s,`latency_avarage`= %(latency_avarage)s, `press_standard_deviation`= %(press_standard_deviation)s,`latency_standard_deviation`= %(latency_standard_deviation)s " 
		" WHERE `id_user`= %(id_user)s AND `id_string`= %(id_string)s");
	executeSqlInDB(updateNormalizedRecord, dbModel);
	
def executeSqlInDB(query, queryModel):	
	try:
		cnx = mysql.connector.connect(user='root', database='moodle');
		cursor = cnx.cursor();
		cursor.execute(query, queryModel);
		cnx.commit();
		cursor.close();
		cnx.close();
		return cursor.lastrowid;
	except mysql.connector.Error as err:
		print(err.msg);
		exit(1);

def getModelQuery(keyModel, idString):
	data_model_keystroke = {
		 'key_code': keyModel.keyCode,
		  'time': keyModel.time,
		  'action': keyModel.action,
		  'id_string': idString,
		}
	return data_model_keystroke;

def getDate(dateMilliseconds): 
	timeMilliseconds = timedelta(milliseconds = dateMilliseconds);
	initialDate = datetime.datetime(1970,1,1); #dates in milliseconds start on (1970,1,1)
	return initialDate + timeMilliseconds;
def convertToMilliseconds(microseconds):
	return microseconds / 1000;


@app.route('/test', methods=['POST', 'GET'])
def test():
	#model = json.loads(request.form['model'], object_hook = keystrokeDecoder);
	#idUser = 21;
	validateKeys = ValidateKeys();
	#validateKeys.predict(idUser, model);
	validateKeys.predictFromDB();
	return "OK";
	
	
class KeystrokeModel:
	def __init__(self, keyCode, time, action=None):
		self.keyCode = keyCode;
		self.time = time;
		self.action = action;
class KeysNormalizedModel:
	def __init__(self, pressAverage, latencyAverage, pressStandardDeviation, latencyStandardDeviation):
		self.pressAverage = pressAverage;
		self.latencyAverage = latencyAverage;
		self.pressStandardDeviation = pressStandardDeviation;
		self.latencyStandardDeviation = latencyStandardDeviation;
class ValidateModel: #remove
	def __init__(self, keyTimePressed, latency): #latency up-down
		self.keyTimePressed = keyTimePressed;
		self.latency = latency;

def keystrokeDecoder(obj):
	return KeystrokeModel(obj['keyCode'], obj['time'], obj['action']);

class Encoder(JSONEncoder):
    def default(self, o):
        return o.__dict__  

class KeystrokeDimensionsExtractor:
	def __init__(self, keys):
		self.keys = keys;
	
	def getTimePressed(self):
		result = [];
		
		for i in range(0, len(self.keys), 1):
			if self.keys[i].action == "DOWN":
				keyUp = self.findKeyUp(self.keys, i+1, self.keys[i]);
				timePressed = getDate(keyUp.time) - getDate(self.keys[i].time);
				result.append(convertToMilliseconds(timePressed.microseconds));
				
		return result;	
	
	def getLatencies(self):
		result = [];
		
		i = 0;
		while i < len(self.keys): #actually, the break will decide when stop
			nextPosition = i + 1;
			keyUp = self.findKeyUp(self.keys, nextPosition, self.keys[i]);
			nextKeyDownPosition = self.findNextKeyDownPosition(self.keys, nextPosition, self.keys[i]);
			if nextKeyDownPosition != None:
				result.append(self.calcLatency(keyUp, self.keys[nextKeyDownPosition])); 
				i = nextKeyDownPosition;
			else:
				break;
				
		return result;
	
	def findKeyUp(self, keys, startPosition, keyDown):
		keyUp = None
		for i in range(startPosition, len(keys), 1):
			if keys[i].action == "UP" and (keys[i].keyCode == keyDown.keyCode):
				keyUp = keys[i]
				break
		return keyUp;
	
	def findNextKeyDownPosition(self, keys, startPosition, keyDown):
		nextKeyDownPosition = None;
		for i in range(startPosition, len(keys), 1):
			if keys[i].action == "DOWN":
				nextKeyDownPosition = i;
				break;
		return nextKeyDownPosition;
	
	def calcLatency(self, keyUp, keyDown):
		latency = None;
		if getDate(keyDown.time) > getDate(keyUp.time):
			latency = getDate(keyDown.time) - getDate(keyUp.time);
			latency = convertToMilliseconds(latency.microseconds);
		else:
			latency = getDate(keyUp.time) - getDate(keyDown.time);
			latency = -convertToMilliseconds(latency.microseconds);
		
		return latency;
		
		
class ValidateKeys:
	#def __init__(self):
	
	def predictFromDB(self):
		users = self.selectUsers();
		print(users);
		for idUser in users:
			originalNormalizedModels = self.selectOriginalModelsFromDB(idUser);
			fakeNormalizedModels = self.selectFakeModelsFromDB(idUser);
			
			fakeTestGroup = self.selectFakeTestGroup(idUser); 
			originalTestGroup = self.selectOriginalTestGroup(idUser); 
			print (originalTestGroup);
			
			validator = ValidateSet(originalNormalizedModels, fakeNormalizedModels);
			validator.knnPredict(fakeTestGroup, False);
			validator.knnPredict(originalTestGroup, True);
		
		print("--------------")
		return "OK" 
		
	def selectUsers(self):		
		result = [];
		try:
			cnx = mysql.connector.connect(user='root', database='moodle')
			cursor = cnx.cursor();
			
			query = ("SELECT `id_user` FROM `mdl_user_key_string` GROUP BY `id_user`");
			
			cursor.execute(query);
			for id_user in cursor.fetchall():
				result.append(id_user[0]);

			cursor.close();
			cnx.close();
		except mysql.connector.Error as err:
			print(err.msg);
			
		return result;
	
	def selectOriginalModelsFromDB(self, idUser):		
		query = ("SELECT `press_average`, `latency_avarage`, `press_standard_deviation`, `latency_standard_deviation`"
				 " FROM mdl_user_keystroke_normalized" 
				 " WHERE id_user = %(idUser)s");
		return self.selectModelsFromDB(query, idUser, self.vectorSerialize);
	def selectOriginalTestGroup(self, idUser):		
		query = ("SELECT `press_average`, `latency_avarage`, `press_standard_deviation`, `latency_standard_deviation`"
				 " FROM mdl_user_test_keystroke_normalized" 
				 " WHERE id_user = %(idUser)s");
		return self.selectModelsFromDB(query, idUser, self.vectorSerialize);

	def selectFakeModelsFromDB(self, idUser):
		query = ("SELECT `press_average`, `latency_avarage`, `press_standard_deviation`, `latency_standard_deviation`"
				" FROM mdl_user_keystroke_normalized" 
				" WHERE id_user <> %(idUser)s");
		return self.selectModelsFromDB(query, idUser, self.vectorSerialize);
	def selectFakeTestGroup(self, idUser):
		query = ("SELECT `press_average`, `latency_avarage`, `press_standard_deviation`, `latency_standard_deviation`"
				" FROM mdl_user_test_keystroke_normalized" 
				" WHERE id_user <> %(idUser)s");
		return self.selectModelsFromDB(query, idUser, self.vectorSerialize);
	
	def vectorSerialize(self, pressAverage, latencyAvarage, pressStandardDeviation, latencyStandardDeviation):
		return [pressAverage, latencyAvarage, pressStandardDeviation, latencyStandardDeviation]
	
	def selectModelsFromDB(self, query, idUser, serielizeCallback):
		models = []
		try:
			cnx = mysql.connector.connect(user='root', database='moodle')
			cursor = cnx.cursor();
			
			cursor.execute(query, {'idUser': idUser});
			for (press_average, latency_avarage, press_standard_deviation, latency_standard_deviation) in cursor:
				models.append(serielizeCallback(press_average, latency_avarage, press_standard_deviation, latency_standard_deviation));

			cursor.close();
			cnx.close();
		except mysql.connector.Error as err:
			print(err.msg);
		return models;
	


class ValidateSet:
	def __init__(self, originalSet, fakeSet):
		self.trainingGroup = (originalSet + fakeSet);
		self.trainingGroupClassification = self.getGroupClassification(originalSet, fakeSet);
		
	def getGroupClassification(self, originalSet, fakeSet):
		result = [];
	
		for key in originalSet:
			result.append("T");
		for key in fakeSet:
			result.append("F");
		
		return result
	
	def knnPredict(self, testDatas, isOriginal = False):
		neigh = KNeighborsClassifier(n_neighbors = 3, metric="euclidean");
		neigh.fit(self.trainingGroup, self.trainingGroupClassification);
		result = neigh.predict(testDatas);
		accuracy = accuracy_score(self.createClass(len(testDatas),isOriginal), result);
		print(result);
		print("accuracy: {:7.2%}".format(accuracy))
		return ''.join(result);
	
	def createClass(self, dataLength, dataIsOriginal):
		result = [];
		label = "F";
		if dataIsOriginal == True:
			label = "T";
		for i in range(0, dataLength, 1):
			result.append(label);
		return result;
	

		

	






