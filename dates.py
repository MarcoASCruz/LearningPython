import datetime
from datetime import timedelta


#millis = int(round(time.time() * 1000))
#print(millis);
#print (datetime.datetime(2008, 9, 5, 11, 25, 38, 498814));
d = timedelta(milliseconds=1468848882265);
inicialDate = datetime.datetime(1970,1,1);
print(inicialDate);
print(d);
print(inicialDate+d);