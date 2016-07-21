from flask import Flask, request, jsonify
from json import JSONEncoder
import json

from datetime import timedelta
import datetime

from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
originalModels = [];
fakeModels = [];

	
@app.route('/validateModel', methods=['POST', 'GET'])
def validateModel():
	global originalModels;
	global fakeModels;
	
	if request.method == 'POST':
		formModel = request.form['model'];
		if (request.form['training'] == "true"):
			if (request.form['original'] == "true"):
				originalModels.append(json.loads(formModel, object_hook = keystrokeDecoder));
			else:
				fakeModels.append(json.loads(formModel, object_hook = keystrokeDecoder));
		else:
			validator = ValidateKeys(originalModels, fakeModels);
			keys = json.loads(formModel, object_hook = keystrokeDecoder);
			validated = validator.validate(keys);
			return Encoder().encode({"originals":validator.originalTimePressedKeys, "fakes":validator.fakeTimePressedKeys, "validated": validated});
	validator = ValidateKeys(originalModels, fakeModels);
	return Encoder().encode({"originals":originalModels, "latency":validator.originalLatency});

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
	def __init__(self, originalKeysMatrix, fakeKeysMatrix):
		self.originalTimePressedKeys = self.getTimePressedKeys(originalKeysMatrix); #originalTimePressedKeys is a matrix too [[key, time]]
		self.fakeTimePressedKeys = self.getTimePressedKeys(fakeKeysMatrix);
		
		self.originalLatency = self.getLatencies(originalKeysMatrix);
		self.fakeLatency = self.getLatencies(fakeKeysMatrix);
		
		self.trainingGroupPressedKey = [];
		self.groupClassificationPressedKey = [];
		
		self.trainingGroupLatency = [];
		self.groupClassificationLatency = [];
		
	def getTimePressedKeys(self, keysMatrix):
		result = [];
		for keys in keysMatrix:
			result.append(self.getTimePressedKey(keys));
		return result;
	def getTimePressedKey(self, keystrokes):
		result = [];
		
		for i in range(0, len(keystrokes), 1):
			if keystrokes[i].action == "DOWN":
				keyUp = self.findKeyUp(keystrokes, i+1, keystrokes[i]);
				timePressed = getDate(keyUp.time) - getDate(keystrokes[i].time);
				keyModel = [keystrokes[i].keyCode, convertToMilliseconds(timePressed.microseconds)];
				result.append(keyModel);
				
		return result;
	
	def getLatencies(self, keysMatrix):
		result = [];
		for keys in keysMatrix:
			result.append(self.getLatency(keys));
		return result;
	def getLatency(self, keystrokes):
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
		
		predictUsingPressedKey = self.predict(self.trainingGroupPressedKey, self.groupClassificationPressedKey, self.getTimePressedKey(keys));
		predictUsingLatency = self.predict(self.trainingGroupLatency, self.groupClassificationLatency, self.getLatency(keys));
		
		return ValidateModel(predictUsingPressedKey, predictUsingLatency);
		
	def createTrainingGroupPressedKey(self): #fill self.trainingGroupPressedKey and groupClassificationPressedKey
		trainingGroupPressedKey = [];
		groupClassificationPressedKey = [];
		for keys in self.originalTimePressedKeys:
			for keyModel in keys:
				trainingGroupPressedKey.append(keyModel);
				groupClassificationPressedKey.append("T");
		for keys in self.fakeTimePressedKeys:
			for keyModel in keys:
				trainingGroupPressedKey.append(keyModel);	
				groupClassificationPressedKey.append("F");
		self.trainingGroupPressedKey = trainingGroupPressedKey;
		self.groupClassificationPressedKey = groupClassificationPressedKey;
		
	def createTrainingGroupLatency(self): #
		trainingGroupLatency = [];
		groupClassificationLatency = [];
		for keys in self.originalLatency:
			for keyModel in keys:
				trainingGroupLatency.append(keyModel);
				groupClassificationLatency.append("T");
		for keys in self.fakeLatency:
			for keyModel in keys:
				trainingGroupLatency.append(keyModel);	
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

	






