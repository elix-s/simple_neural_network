using UnityEngine;

public class SimpleAgent : MonoBehaviour
{
    [SerializeField] private Transform _target;
    [SerializeField] private float _moveSpeed = 5f;
    [SerializeField] private float _rewardThreshold = 0.5f;  
    [SerializeField] private float _maxDistance = 10f;       

    private SimpleNeuralNetwork _neuralNetwork;
    private Vector2 _startPosition;       
    private Vector2 _targetStartPosition;  
    private float _resetTime = 15f;         
    private float _timer;                
    private float _lastDistanceToTarget;   

    private void Start()
    {
        _neuralNetwork = new SimpleNeuralNetwork(1, 8, 1);
        
        _startPosition = transform.localPosition;
        _targetStartPosition = _target.localPosition;
        _timer = _resetTime;
        _lastDistanceToTarget = Mathf.Abs(transform.localPosition.x - _target.localPosition.x);
    }

    private void Update()
    {
        _timer -= Time.deltaTime;
        
        if (_timer <= 0f)
        {
            ResetAgentAndTarget();
            _timer = _resetTime;  
        }
        
        float distanceX = transform.localPosition.x - _target.localPosition.x;
        float normalizedInput = Mathf.Clamp(distanceX / _maxDistance, -1f, 1f);

        float[] inputs = new float[1] { normalizedInput };
        float[] actions = _neuralNetwork.FeedForward(inputs);
        
        float speedMultiplier = Mathf.Max(1f, Mathf.Abs(distanceX)); 
        float movementX = Mathf.Clamp(actions[0], -1f, 1f) * _moveSpeed * speedMultiplier * Time.deltaTime;
        
        if (Mathf.Abs(movementX) < 0.001f)
        {
            movementX = UnityEngine.Random.Range(-0.1f, 0.1f) * _moveSpeed * Time.deltaTime;
        }
        
        if (float.IsNaN(movementX))
        {
            movementX = 0f;
        }
        
        transform.Translate(new Vector2(movementX, 0));
        
        float distanceToTarget = Mathf.Abs(transform.localPosition.x - _target.localPosition.x);
        float[] targetOutputs = new float[1];
        
        if (distanceToTarget < _lastDistanceToTarget)
        {
            targetOutputs[0] = 0.5f; 
        }
        else
        {
            targetOutputs[0] = -0.5f; 
        }
        
        if (distanceToTarget < _rewardThreshold)
        {
            targetOutputs[0] = 1f; 
            _neuralNetwork.Train(inputs, targetOutputs);
            Debug.Log("Target Reached!");
            ResetAgentAndTarget();
        }
        else
        {
            _neuralNetwork.Train(inputs, targetOutputs);
        }
        
        _lastDistanceToTarget = distanceToTarget;
    }
    
    private void ResetAgentAndTarget()
    {
        transform.localPosition = _startPosition;
        _target.localPosition = _targetStartPosition;
        _lastDistanceToTarget = Mathf.Abs(transform.localPosition.x - _target.localPosition.x);
    }
}
