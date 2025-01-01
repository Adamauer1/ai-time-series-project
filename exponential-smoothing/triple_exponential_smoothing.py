# s = a(y-cp) + (1-a)(sp+bp)
# b = be(s-sp) + (1-be)bp
# c = g(y-s) + (1-g)cp
# f = s + b + c*m
# ff = s + hb + c

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
class TripleExponentialSmoothing:
    def __init__ (self, alpha, beta, gamma):
        self.data = None
        self.s = None
        self.b = None
        self.c = None
        self.c_length = 0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def fit (self, data, c_length):
        self.data = data;
        self.s = [data[0]]
        self.b = [data[1] - data[0]]
        self.c_length = c_length;
        self.c = []
        data_smoothed = []
        for i in range(self.c_length):
            self.c.append(data[i] - self.s[0])

        for i in range(1, len(data)):
            c_index = (i - self.c_length) % self.c_length
            # first / simple
            self.s.append(self.alpha*(data[i-1] - self.c[c_index]) + (1-self.alpha)*(self.s[i-1] + self.b[i-1]))
            # second / trend
            self.b.append(self.beta * (self.s[i] - self.s[i-1]) + (1 - self.beta) * self.b[i-1])
            # third / season
            self.c[i % self.c_length] = self.gamma * (data[i] - self.s[i]) + (1 - self.gamma) * self.c[c_index]
            #self.c.append(self.gamma * (data[i] - self.s[i]) + (1 - self.gamma) * self.c[c_index])
            data_smoothed.append(self.s[i] + self.b[i] + self.c[i % self.c_length])
        return data_smoothed


    def predict(self, steps=1):
        forcast = []
        for i in range(1, steps+1):
            #c_index = (len(self.data)-self.c_length+(i-1)) % self.c_length
            c_index = (len(self.data)-self.c_length+(i-1)) % self.c_length
            forcast.append(self.s[-1] + i * self.b[-1] + self.c[c_index])
        return forcast


df = pd.read_csv("../DailyDelhiClimateTrain.csv")

data = df["meantemp"].dropna().to_numpy()

dfT = pd.read_csv("../DailyDelhiClimateTest.csv")

realData = dfT["meantemp"].dropna().to_numpy()

dates = df["date"]

dates = pd.to_datetime(dates)

alpha = 0.5
beta = 0.3
gamma = 0.7

model = TripleExponentialSmoothing(alpha, beta, gamma)

smoothed_values = model.fit(data, 365)
forecasts = model.predict(114)


plt.figure(figsize=(12, 6))
# plt.plot(data, label="Original", marker="o")
# plt.plot(smoothed_values+forecasts, label="Triple", linestyle="--", marker="x")
plt.plot(forecasts, label="Triple", linestyle="--", marker="x")
plt.plot(realData, label="Real", marker="o")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()