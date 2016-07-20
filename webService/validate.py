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
		
		self.trainingGroup = [];
		self.groupClassification = [];
		
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
					"{0} and {1}".format(keyUp.keyCode, keystrokes[nextKeyDownPosition].keyCode)
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

		print(getDate(keyUp.time));
		print(getDate(keyDown.time));
		
		return latency;
	
	def validate(self, keys):
		self.createTrainingGroup();
		neigh = KNeighborsClassifier(n_neighbors = 3);
		neigh.fit(self.trainingGroup, self.groupClassification);
		print(self.getTimePressedKey(keys));
		return ''.join(neigh.predict(self.getTimePressedKey(keys)));
	def createTrainingGroup(self): #fill self.trainingGroup and groupClassification
		trainingGroup = [];
		groupClassification = [];
		for keys in self.originalTimePressedKeys:
			for keyModel in keys:
				trainingGroup.append(keyModel);
				groupClassification.append("T");
		for keys in self.fakeTimePressedKeys:
			for keyModel in keys:
				trainingGroup.append(keyModel);	
				groupClassification.append("F");
		self.trainingGroup = trainingGroup;
		self.groupClassification = groupClassification;

def getDate(dateMilliseconds): 
	timeMilliseconds = timedelta(milliseconds = dateMilliseconds);
	initialDate = datetime.datetime(1970,1,1); #dates in milliseconds start on (1970,1,1)
	return initialDate + timeMilliseconds;
def convertToMilliseconds(microseconds):
	return microseconds / 1000;

	






