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
        self.data = data
        self.s = [data[0]]
        self.b = [data[1] - data[0]]
        self.c_length = c_length
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
            self.c[i % self.c_length] = self.gamma * (data[i] - self.s[i-1] - self.b[i-1]) + (1 - self.gamma) * self.c[c_index]
            data_smoothed.append(self.s[i] + self.b[i] + self.c[i % self.c_length])
        return data_smoothed

    def predict(self, steps=1):
        forcast = []
        for i in range(1, steps+1):
            c_index = (len(self.data)-self.c_length+(i-1)) % self.c_length
            forcast.append(self.s[-1] + i * self.b[-1] + self.c[c_index])
        return forcast
