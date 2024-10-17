using UnityEngine;
using Random = System.Random;

public class SimpleNeuralNetwork
{
    private int _inputSize;
    private int _hiddenSize;
    private int _outputSize;
    private float[,] _weights1;
    private float[] _biases1;
    private float[] _hiddenLayer;
    private float[] _outputLayer;

    public SimpleNeuralNetwork(int inputSize, int hiddenSize, int outputSize)
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _outputSize = outputSize;

        // Инициализация весов и смещений случайными значениями
        _weights1 = new float[inputSize, hiddenSize];
        _biases1 = new float[hiddenSize];
        _hiddenLayer = new float[hiddenSize];
        _outputLayer = new float[outputSize];

        InitializeWeights();
    }

    // Инициализация весов случайными значениями 
    private void InitializeWeights()
    {
        Random rand = new Random();

        for (int i = 0; i < _inputSize; i++)
        {
            for (int j = 0; j < _hiddenSize; j++)
            {
                _weights1[i, j] = (float)(rand.NextDouble() * 4.0 - 2.0);
            }
        }

        for (int i = 0; i < _hiddenSize; i++)
        {
            _biases1[i] = (float)(rand.NextDouble() * 4.0 - 2.0);  // От -2 до 2
        }
    }

    // Прямая передача данных через нейросеть
    public float[] FeedForward(float[] inputs)
    {
        for (int i = 0; i < _hiddenSize; i++)
        {
            _hiddenLayer[i] = 0f;

            for (int j = 0; j < _inputSize; j++)
            {
                _hiddenLayer[i] += inputs[j] * _weights1[j, i];
            }

            _hiddenLayer[i] += _biases1[i];
            _hiddenLayer[i] = ReLU(_hiddenLayer[i]); 
        }

        // Выходной слой
        for (int i = 0; i < _outputSize; i++)
        {
            _outputLayer[i] = _hiddenLayer[i];  
        }

        return _outputLayer;
    }

    // Обратное распространение ошибки и корректировка весов
    public void Train(float[] inputs, float[] targets)
    {
        float learningRate = 0.05f;  

        for (int i = 0; i < _hiddenSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                float output = FeedForward(inputs)[0];
                float error = targets[0] - output;

                _weights1[j, i] += learningRate * error * inputs[j]; 
            }
        }
    }
    
    private float ReLU(float x)
    {
        return Mathf.Max(0, x);
    }
}
