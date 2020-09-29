import math
import random
import json


class Neuron:
    __count_of_inputs: int
    __inputs = []
    __weights = []
    __t_value: float

    def __init__(self, count_of_inputs: int):
        if count_of_inputs <= 0:
            raise ValueError("count of inputs in neuron can be biggest zero")

        self.__count_of_inputs = count_of_inputs

        for i in range(self.__count_of_inputs):
            self.__inputs.append(0.0)
            self.__weights.append(0.0)

    def GetCountOfInputs(self):
        return self.__count_of_inputs

    def SetInput(self, index: int, value: float):
        if index < 0 or index >= self.__count_of_inputs:
            raise ValueError("index of input can be biggest or equals zero and less count of inputs")

        self.__inputs[index] = value

    def GetInput(self, index: int):
        if index < 0 or index >= self.__count_of_inputs:
            raise ValueError("index of input can be biggest or equals zero and less count of inputs")

        return self.__inputs[index]

    def SetWeight(self, index: int, value: float):
        if index < 0 or index >= self.__count_of_inputs:
            raise ValueError("index of weight in neuron can be biggest  or equals zero and less count of weight")

        self.__weights[index] = value

    def GetWeight(self, index: int):
        if index < 0 or index >= self.__count_of_inputs:
            raise ValueError("index of weight in neuron can be biggest  or equals zero and less count of weight")

        return self.__weights[index]

    def SetTValue(self, t):
        self.__t_value = t

    def GetTValue(self) -> float:
        return self.__t_value

    def Calculate(self) -> float:
        result = 0.0

        for i in range(self.__count_of_inputs):
            temp = self.__weights[i] * self.__inputs[i]
            result += temp

        return result

    @staticmethod
    def SigmoidActivation(value: float) -> float:
        return 1.0 / (1.0 + math.exp(value))

    @staticmethod
    def StepActivation(value: float) -> float:
        if value > 0:
            return 1.0
        else:
            return -1.0


NEEDABLE_ERROR = 0.25


class HebbLearningAlgorithm:
    __learning_data_set_count: int
    __learning_data_set = []
    __learning_data_set_result = []

    __trainingNeuron: Neuron

    def __init__(self):
        self.__learning_data_set_count = 0

    def SetLearningDataSet(self, data_set, result):
        self.__learning_data_set_count += 1
        self.__learning_data_set.append(data_set)
        self.__learning_data_set_result.append(result)

    def SetNeuron(self, neuron: Neuron):
        self.__trainingNeuron = neuron

    def GetNeuron(self) -> Neuron:
        return self.__trainingNeuron

    def SetRandomWeights(self):
        for i in range(self.__trainingNeuron.GetCountOfInputs()):
            self.__trainingNeuron.SetWeight(i, random.uniform(-0.5, 0.5))

    def Learning(self):
        self.SetRandomWeights()

        needLoop = True
        while needLoop:
            haveAError = False
            for i in range(self.__learning_data_set_count):
                t = 0
                for item in self.__learning_data_set[i]:
                    self.__trainingNeuron.SetInput(t, item)
                    t += 1

                result = self.__trainingNeuron.Calculate()
                result_activation = Neuron.StepActivation(result)

                if result_activation != self.__learning_data_set_result[i]:
                    haveAError = True

                    for k in range(self.__trainingNeuron.GetCountOfInputs()):
                        _input = self.__trainingNeuron.GetInput(k)
                        _weight = self.__trainingNeuron.GetWeight(k)
                        new_weight = _weight - result_activation * _input
                        self.__trainingNeuron.SetWeight(k, new_weight)

            if not haveAError:
                needLoop = False

    def SaveDataset(self, file_name: str):
        save_data = []

        for i in range(self.__learning_data_set_count):
            temp = {
                "input": self.__learning_data_set[i],
                "result": self.__learning_data_set_result[i]
            }

            save_data.append(temp)

        file_content = json.dumps(save_data)

        fp = open(file_name, "w")
        fp.write(file_content)
        fp.close()

    def LoadDataset(self, file_name: str):
        self.__learning_data_set_count = 0
        self.__learning_data_set.clear()
        self.__learning_data_set_result.clear()

        fp = open(file_name, "r")
        file_content = fp.readline()
        fp.close()

        load_data = json.loads(file_content)

        for item in load_data:
            self.__learning_data_set_count += 1
            self.__learning_data_set.append(item["input"])
            self.__learning_data_set_result.append(item["result"])


def learning(neuron: Neuron):
    hebb = HebbLearningAlgorithm()

    hebb.LoadDataset("data_set.json")

    hebb.SetNeuron(neuron)

    hebb.Learning()
    neuron = hebb.GetNeuron()

    for i in range(neuron.GetCountOfInputs()):
        print("weight index = " + str(i) + " value = " + str(neuron.GetWeight(i)))


def main():
    test_data = [-1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1]

    neuron = Neuron(15)
    learning(neuron)

    for i in range(len(test_data)):
        neuron.SetInput(i, test_data[i])

    result = neuron.StepActivation(neuron.Calculate())
    print(result)


if __name__ == "__main__":
    main()
