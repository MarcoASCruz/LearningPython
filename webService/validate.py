from flask import Flask, request, jsonify
from json import JSONEncoder
import json

import mysql.connector

from datetime import timedelta
import datetime

from sklearn.neighbors import KNeighborsClassifier

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
	print(idUser);
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
	
	for model in models: # para cada string
		string = createStringFromKeys(model); # for improve the performance I can update the string after the for bellow
		idString = insertInDB(insertString, (idUser, string));
		for key in model:
			insertInDB(insertKeystroke, getModelQuery(key, idString))

def createStringFromKeys(keys):
	string = "";
	for key in keys:
		string = string + '{}_'.format(key.keyCode);
	return string[: len(string) - 1];
	
def insertInDB(query, queryModel):	
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

def selectOriginalModelsFromDB(idUser):		
	query = "SELECT keyCode, time, action FROM mdl_user_info_keystroke WHERE idUser = %(idUser)s ORDER BY time";
	return selectModelsFromDB(query, idUser);

def selectFakeModelsFromDB(idUser):
	query = "SELECT keyCode, time, action FROM mdl_user_info_keystroke WHERE idUser <> %(idUser)s ORDER BY time";
	return selectModelsFromDB(query, idUser);

def selectModelsFromDB(query, idUser):
	models = []
	try:
		cnx = mysql.connector.connect(user='root', database='moodle')
		cursor = cnx.cursor();
		
		cursor.execute(query, {'idUser': idUser});
		for (keyCode, time, action) in cursor:
			models.append(KeystrokeModel(keyCode, time, action));

		cursor.close();
		cnx.close();
	except mysql.connector.Error as err:
		print(err.msg);
	return models;
	
class KeystrokeModel:
	def __init__(self, keyCode, time, action=None):
		self.keyCode = keyCode;
		self.time = time;
		self.action = action;
class ValidateModel:
	def __init__(self, keyTimePressed, latency): #latency up-down
		self.keyTimePressed = keyTimePressed;
		self.latency = latency;

def keystrokeDecoder(obj):
	return KeystrokeModel(obj['keyCode'], obj['time'], obj['action']);

class Encoder(JSONEncoder):
    def default(self, o):
        return o.__dict__  

class ValidateKeys:		
	def __init__(self, originalKeys, fakeKeys):
		self.originalTimePressedKeys = self.getTimePressedKeys(originalKeys);
		self.fakeTimePressedKeys = self.getTimePressedKeys(fakeKeys);
		
		self.originalLatency = self.getLatencies(originalKeys);
		self.fakeLatency = self.getLatencies(fakeKeys);
		
		self.trainingGroupPressedKey = [];
		self.groupClassificationPressedKey = [];
		
		self.trainingGroupLatency = [];
		self.groupClassificationLatency = [];
		
	def getTimePressedKeys(self, keystrokes):
		result = [];
		
		for i in range(0, len(keystrokes), 1):
			if keystrokes[i].action == "DOWN":
				keyUp = self.findKeyUp(keystrokes, i+1, keystrokes[i]);
				timePressed = getDate(keyUp.time) - getDate(keystrokes[i].time);
				keyModel = [keystrokes[i].keyCode, convertToMilliseconds(timePressed.microseconds)];
				result.append(keyModel);
				
		return result;
	
	def getLatencies(self, keystrokes):
		result = [];
		
		i = 0;
		while i < len(keystrokes): #actually, the break will decide when stop
			nextPosition = i + 1;
			keyUp = self.findKeyUp(keystrokes, nextPosition, keystrokes[i]);
			nextKeyDownPosition = self.findNextKeyDownPosition(keystrokes, nextPosition, keystrokes[i]);
			if nextKeyDownPosition != None:
				keyModel = [
					"{0}.{1}".format(keyUp.keyCode, keystrokes[nextKeyDownPosition].keyCode)
					,
					self.calcLatency(keyUp, keystrokes[nextKeyDownPosition])
				];
				result.append(keyModel); 
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
	
	def validate(self, keys):
		self.createTrainingGroupPressedKey();
		self.createTrainingGroupLatency();
		
		print(self.originalTimePressedKeys)
		print("-------------------")
		print(self.trainingGroupPressedKey)
		print("-------------------")
		print(self.groupClassificationPressedKey)
		print("-------------------")
		print(self.trainingGroupLatency)
		print("-------------------")
		print(self.groupClassificationLatency)
		print("-------------------")
		print(self.getTimePressedKeys(keys))
		print("-------------------")
		print(self.getLatencies(keys))
		
		predictUsingPressedKey = self.predict(self.trainingGroupPressedKey, self.groupClassificationPressedKey, self.getTimePressedKeys(keys));
		predictUsingLatency = self.predict(self.trainingGroupLatency, self.groupClassificationLatency, self.getLatencies(keys));
		
		return ValidateModel(predictUsingPressedKey, predictUsingLatency);
		
	def createTrainingGroupPressedKey(self): #fill self.trainingGroupPressedKey and groupClassificationPressedKey
		trainingGroupPressedKey = [];
		groupClassificationPressedKey = [];
		for key in self.originalTimePressedKeys:
			trainingGroupPressedKey.append(key);
			groupClassificationPressedKey.append("T");
		for key in self.fakeTimePressedKeys:	
			trainingGroupPressedKey.append(key);
			groupClassificationPressedKey.append("F");
		self.trainingGroupPressedKey = trainingGroupPressedKey;
		self.groupClassificationPressedKey = groupClassificationPressedKey;
		
	def createTrainingGroupLatency(self): #
		trainingGroupLatency = [];
		groupClassificationLatency = [];
		for key in self.originalLatency:
			trainingGroupLatency.append(key);
			groupClassificationLatency.append("T");
		for key in self.fakeLatency:
			trainingGroupLatency.append(key);
			groupClassificationLatency.append("F");
		self.trainingGroupLatency = trainingGroupLatency;
		self.groupClassificationLatency = groupClassificationLatency;
	def predict(self, trainingGroup, groupClassification, testDatas):
		neigh = KNeighborsClassifier(n_neighbors = 3, metric="euclidean");
		neigh.fit(trainingGroup, groupClassification);
		return ''.join(neigh.predict(testDatas));

def getDate(dateMilliseconds): 
	timeMilliseconds = timedelta(milliseconds = dateMilliseconds);
	initialDate = datetime.datetime(1970,1,1); #dates in milliseconds start on (1970,1,1)
	return initialDate + timeMilliseconds;
def convertToMilliseconds(microseconds):
	return microseconds / 1000;

	






