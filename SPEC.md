# CrossPhaseMiner - Technical Specification

## 1. Project Overview
CrossPhaseMiner is a signal cycle learning system for delivery robots. It learns pedestrian crossing signal patterns from visual observations and assists crossing decisions.

## 2. Architecture

```
crossphase_miner/
├── core/
│   ├── __init__.py
│   ├── models.py              # Core data structures (dataclasses)
│   ├── period_learner.py      # Bayesian + UKF cycle parameter estimation
│   ├── tod_manager.py         # Time-of-Day multi-period management
│   ├── predictor.py           # LSTM/TCN temporal prediction
│   └── decision_fusion.py     # Vision + cycle model fusion
├── simulation/
│   ├── __init__.py
│   ├── data_generator.py      # Realistic Korean signal simulation
│   └── visualizer.py          # Plotting and visualization
├── utils/
│   ├── __init__.py
│   └── math_utils.py          # Gaussian operations, circular stats
├── tests/
│   └── test_all.py            # Unit and integration tests
├── main.py                    # End-to-end simulation demo
└── requirements.txt
```

## 3. Core Data Models (models.py)

### SignalColor (Enum)
- RED, GREEN, FLASHING_GREEN, UNKNOWN

### SignalObservation
```python
@dataclass
class SignalObservation:
    intersection_id: str          # e.g., "crosswalk_001"
    timestamp: float              # Unix timestamp (seconds)
    color: SignalColor            # Observed signal color
    confidence: float             # Vision model confidence [0, 1]
    robot_id: str                 # Source robot identifier
```

### PhaseTransition
```python
@dataclass  
class PhaseTransition:
    intersection_id: str
    timestamp: float              # Transition time
    from_color: SignalColor
    to_color: SignalColor
```

### PeriodModel
```python
@dataclass
class PeriodModel:
    T_cycle: float                # Cycle period (seconds)
    T_red: float                  # Red phase duration
    T_green: float                # Green phase duration
    phi_offset: float             # Phase offset (seconds from day start)
    confidence: float             # Model confidence [0, 1]
    sample_count: int             # Number of observations used
    last_updated: float           # Timestamp
    
    # Methods
    predict_next_green(current_time: float) -> Tuple[float, float]
    predict_remaining_time(current_time: float, current_color: SignalColor) -> float
```

### IntersectionProfile
```python
@dataclass
class IntersectionProfile:
    intersection_id: str
    tod_models: Dict[str, PeriodModel]  # key: "morning_rush", "off_peak", etc.
    is_push_button: bool = False
    is_learnable: bool = True
    total_observations: int = 0
    creation_time: float = 0.0
```

## 4. Period Learner (period_learner.py)

### BayesianPeriodLearner
- **Purpose**: Online Bayesian estimation of signal cycle parameters
- **State**: Prior distributions over T_cycle, T_red, phi_offset
- **Update**: Each new PhaseTransition updates posterior via Bayesian update
- **Output**: PeriodModel with confidence intervals

Key methods:
```python
class BayesianPeriodLearner:
    def __init__(self, prior_T_cycle: Tuple[float, float] = (120, 30))
    def update(self, transition: PhaseTransition) -> PeriodModel
    def get_model(self) -> PeriodModel
    def reset(self) -> None
```

### UKFPeriodLearner  
- **Purpose**: Unscented Kalman Filter for non-linear phase estimation
- **State Vector**: x = [T_cycle, T_red, phi_offset]
- **Process Model**: Constant parameters with small noise
- **Measurement**: Observed transition times

Key methods:
```python
class UKFPeriodLearner:
    def __init__(self, initial_state: np.ndarray, process_noise: float = 0.01)
    def update(self, transition: PhaseTransition, current_time: float) -> PeriodModel
    def predict_transition(self, target_color: SignalColor) -> float
    def get_confidence_ellipse(self) -> np.ndarray
```

## 5. TOD Manager (tod_manager.py)

### TODManager
- Manages 4 default periods: morning_rush, off_peak, evening_rush, night
- Period boundaries: 06:00-08:30, 08:30-16:00, 16:00-21:00, 21:00-06:00
- Auto-detects TOD switches via prediction error monitoring
- Maintains IntersectionProfile per intersection

```python
class TODManager:
    def __init__(self, period_configs: Optional[List[TODPeriodConfig]] = None)
    def classify_tod(self, timestamp: float) -> str
    def get_model(self, intersection_id: str, timestamp: float) -> Optional[PeriodModel]
    def update_model(self, intersection_id: str, timestamp: float, 
                     model: PeriodModel) -> None
    def detect_tod_switch(self, intersection_id: str, 
                          recent_errors: List[float]) -> bool
    def create_intersection(self, intersection_id: str) -> IntersectionProfile
    def should_learn(self, intersection_id: str) -> bool
```

## 6. Predictor (predictor.py)

### SignalPredictor (LSTM-based)
- Input sequence: N observations of (timestamp, color_encoded, confidence)
- Output: predicted_seconds_to_next_green, predicted_remaining_green
- Trainable on accumulated observation history
- ONNX exportable

```python
class SignalPredictor:
    def __init__(self, seq_length: int = 20, hidden_dim: int = 64)
    def fit(self, observations: List[SignalObservation]) -> None
    def predict(self, recent_observations: List[SignalObservation]) -> np.ndarray
    def export_onnx(self, path: str) -> None
    def load_onnx(self, path: str) -> None
```

### TCNPredictor (alternative)
- Temporal Convolutional Network with dilated convolutions
- Same interface as SignalPredictor

## 7. Decision Fusion (decision_fusion.py)

### DecisionFusionEngine
- Hard safety constraints (red = always stop)
- Soft optimization (green + high confidence model = reduce confirmation threshold)
- Push-button detection via variance analysis
- Multi-robot data aggregation

```python
class DecisionFusionEngine:
    def __init__(self, tod_manager: TODManager, 
                 default_confirm_threshold: int = 5)
    
    def make_decision(self, intersection_id: str,
                     current_color: SignalColor,
                     color_confidence: float,
                     current_time: float,
                     safe_crossing_time: float = 8.0) -> DecisionResult
    
    def detect_push_button(self, intersection_id: str,
                          history: List[PhaseTransition]) -> bool
    
    def aggregate_observations(self, all_observations: List[SignalObservation])
                              -> Dict[str, List[SignalObservation]]
```

### DecisionResult
```python
@dataclass
class DecisionResult:
    action: str                    # "WAIT", "GO", "PREPARE", "ABORT"
    confidence: float
    reason: str
    visual_confirm_required: int   # Required consecutive green frames
    predicted_green_time: float    # ETA to next green
    predicted_green_remaining: float
    risk_level: str               # "LOW", "MEDIUM", "HIGH"
```

## 8. Simulation Data Generator (simulation/data_generator.py)

### SignalSimulator
- Simulates Korean TOD signal patterns
- Adds realistic noise (vision delay, misclassification, push-button randomness)
- Generates training/evaluation datasets

```python
class SignalSimulator:
    def __init__(self, random_seed: int = 42)
    
    def add_intersection(self, intersection_id: str, 
                        cycle_configs: Dict[str, tuple],  # tod -> (T_cycle, T_red)
                        is_push_button: bool = False)
    
    def simulate_pass(self, intersection_id: str, 
                     arrival_time: float,
                     robot_id: str = "robot_001") -> List[SignalObservation]
    
    def generate_dataset(self, intersection_id: str,
                        num_passes: int,
                        time_range: Tuple[float, float]) -> List[SignalObservation]
```

Default Korean TOD config:
```python
KOREAN_TOD_DEFAULT = {
    "morning_rush": {"T_cycle": 200, "T_red": 140, "T_green": 60},
    "off_peak":     {"T_cycle": 150, "T_red": 100, "T_green": 50},
    "evening_rush": {"T_cycle": 180, "T_red": 120, "T_green": 60},
    "night":        {"T_cycle": 120, "T_red": 80,  "T_green": 40},
}
```

## 9. Main Simulation (main.py)

End-to-end demonstration:
1. Generate simulated data for 3 intersections (2 regular + 1 push-button)
2. Run period learning (Bayesian + UKF comparison)
3. Build TOD models
4. Make crossing decisions at various arrival times
5. Visualize results

## 10. Dependencies (requirements.txt)
```
numpy>=1.24.0
matplotlib>=3.7.0
torch>=2.0.0
scipy>=1.10.0
onnx>=1.14.0
onnxruntime>=1.15.0
```
