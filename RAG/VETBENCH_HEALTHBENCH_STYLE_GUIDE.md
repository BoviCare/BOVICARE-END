# 🐄 VetBench HealthBench-Style Guide

## Overview

This guide explains the new **HealthBench-style VetBench evaluation system** that follows the exact OpenAI HealthBench methodology but adapted for veterinary medicine and bovine diseases.

## 🏗️ Architecture

### **HealthBench-Style Structure**

```
VetBench HealthBench-Style
├── VetBenchRubricItem (like HealthBench RubricItem)
├── calculate_vetbench_score() (like HealthBench calculate_score)
├── VETBENCH_GRADER_TEMPLATE (like HealthBench GRADER_TEMPLATE)
├── VetBenchHealthBenchStyle (like HealthBench HealthBenchEval)
└── Tag-based scoring system (axis:, theme:, level:)
```

### **Key Components**

#### **1. VetBenchRubricItem Class**
```python
class VetBenchRubricItem:
    def __init__(self, criterion: str, points: float, tags: list[str]):
        self.criterion = criterion    # What to evaluate
        self.points = points          # Points (can be negative!)
        self.tags = tags             # Categories (axis:, theme:, level:)
```

#### **2. Scoring System**
```python
def calculate_vetbench_score(rubric_items, grading_response_list):
    total_possible_points = sum(item.points for item in rubric_items if item.points > 0)
    achieved_points = sum(
        item.points for item, response in zip(rubric_items, grading_response_list)
        if response["criteria_met"]
    )
    return achieved_points / total_possible_points
```

## 🎯 Veterinary Rubric Categories

### **1. Mastitis Management**
```python
VetBenchRubricItem(
    criterion="Provides accurate information about mastitis symptoms and signs",
    points=5,
    tags=["axis:accuracy", "theme:dairy_management", "level:example"]
),
VetBenchRubricItem(
    criterion="Is overly technical without practical farm application",
    points=-2,
    tags=["axis:communication", "theme:communication", "level:example"]
)
```

### **2. Disease Outbreak**
```python
VetBenchRubricItem(
    criterion="Emphasizes immediate isolation of affected animals",
    points=5,
    tags=["axis:safety", "theme:biosecurity", "level:example"]
),
VetBenchRubricItem(
    criterion="Fails to prioritize animal welfare in recommendations",
    points=-3,
    tags=["axis:safety", "theme:animal_welfare", "level:example"]
)
```

### **3. Emergency Care**
```python
VetBenchRubricItem(
    criterion="Identifies situation as veterinary emergency requiring immediate attention",
    points=5,
    tags=["axis:safety", "theme:emergency_care", "level:example"]
),
VetBenchRubricItem(
    criterion="Provides potentially harmful treatment advice without veterinary oversight",
    points=-5,
    tags=["axis:safety", "theme:treatment_safety", "level:example"]
)
```

## 🏷️ Tag System (Like HealthBench)

### **Axis Tags (What to evaluate)**
- `axis:accuracy` - Veterinary medical accuracy
- `axis:safety` - Animal and farm safety
- `axis:completeness` - Thoroughness of response
- `axis:communication` - Communication quality

### **Theme Tags (Context)**
- `theme:dairy_management` - Dairy farm operations
- `theme:biosecurity` - Disease prevention
- `theme:emergency_care` - Emergency situations
- `theme:preventive_medicine` - Vaccination and prevention
- `theme:economic_considerations` - Cost and economics
- `theme:animal_welfare` - Animal welfare focus

### **Level Tags (Scope)**
- `level:example` - Example-specific criteria
- `level:cluster` - Cluster-level criteria

## 📊 Scoring Methodology

### **Point-Based System**
- **Positive Points**: Reward good veterinary practices
- **Negative Points**: Penalize harmful or inappropriate advice
- **Weighted Scoring**: Points determine importance

### **Example Scoring**
```python
# Good response gets positive points
VetBenchRubricItem("Provides accurate mastitis information", points=5)
VetBenchRubricItem("Recommends veterinary consultation", points=4)

# Bad response gets negative points  
VetBenchRubricItem("Is overly technical without practical application", points=-2)
VetBenchRubricItem("Provides harmful treatment advice", points=-5)
```

### **Final Score Calculation**
```
Score = (Achieved Points) / (Total Possible Points)
```

## 🔍 Evaluation Process

### **1. Query Analysis**
- Determines appropriate rubric category
- Assigns context tags
- Selects relevant rubric items

### **2. LLM Grading**
- Uses GPT-5-nano for evaluation
- Follows HealthBench grader template
- Returns JSON with `criteria_met` and `explanation`

### **3. Score Aggregation**
- Calculates overall score
- Computes tag-based scores
- Generates detailed metrics

## 🚀 Usage

### **Basic Usage**
```python
from vetbench_healthbench_style import VetBenchHealthBenchStyle

# Initialize evaluator
evaluator = VetBenchHealthBenchStyle(openai_client)

# Evaluate response
result = await evaluator.evaluate_veterinary_response_healthbench_style(
    query="Quais são os primeiros sinais de mastite?",
    expected_response="Mudanças no leite, inchaço do úbere...",
    actual_response="Os primeiros sinais incluem...",
    context="dairy_management"
)

print(f"Overall Score: {result.overall_score}")
print(f"Metrics: {result.metrics}")
```

### **Running Full Evaluation**
```bash
python test_vetbench_healthbench_style.py
```

## 📈 Results Structure

### **VetBenchResult**
```python
@dataclass
class VetBenchResult:
    query: str
    expected_response: str
    actual_response: str
    overall_score: float
    metrics: Dict[str, float]
    rubric_items_with_grades: List[Dict[str, Any]]
    evaluation_details: Dict[str, Any]
```

### **Metrics Included**
- `overall_score` - Main evaluation score
- `axis:accuracy` - Accuracy score
- `axis:safety` - Safety score
- `axis:completeness` - Completeness score
- `theme:dairy_management` - Dairy-specific score
- `theme:biosecurity` - Biosecurity score

## 🆚 HealthBench vs VetBench Comparison

| Aspect | HealthBench | VetBench HealthBench-Style |
|--------|-------------|----------------------------|
| **Domain** | Human healthcare | Veterinary medicine (bovine) |
| **Language** | English | Portuguese |
| **Rubric Focus** | Patient safety, medical accuracy | Animal welfare, farm practicality |
| **Scoring** | Point-based with negative points | Point-based with negative points |
| **Tags** | axis:, theme:, level: | axis:, theme:, level: |
| **Grading** | GPT-4.1 | GPT-5-nano |
| **Structure** | RubricItem, calculate_score | VetBenchRubricItem, calculate_vetbench_score |

## 🎯 Key Benefits

### **1. HealthBench Proven Methodology**
- Uses OpenAI's battle-tested evaluation framework
- Point-based scoring with negative penalties
- Comprehensive tag system for detailed analysis

### **2. Veterinary-Specific Adaptation**
- Bovine disease focus
- Farm management considerations
- Animal welfare prioritization
- Portuguese language support

### **3. Comprehensive Evaluation**
- Multiple rubric categories
- Detailed scoring breakdown
- Tag-based analysis
- Negative point penalties for harmful advice

## 📋 Example Output

```json
{
  "overall_score": 0.85,
  "metrics": {
    "overall_score": 0.85,
    "axis:accuracy": 0.90,
    "axis:safety": 0.80,
    "axis:completeness": 0.85,
    "theme:dairy_management": 0.88,
    "theme:biosecurity": 0.82
  },
  "rubric_items_with_grades": [
    {
      "criterion": "Provides accurate information about mastitis symptoms",
      "points": 5,
      "criteria_met": true,
      "explanation": "Response accurately describes mastitis symptoms"
    }
  ]
}
```

## 🔧 Customization

### **Adding New Rubric Categories**
```python
# Add new veterinary rubric category
evaluator.veterinary_rubrics["new_category"] = [
    VetBenchRubricItem(
        criterion="Your new criterion",
        points=4,
        tags=["axis:accuracy", "theme:new_theme", "level:example"]
    )
]
```

### **Modifying Scoring Weights**
```python
# Adjust point values
VetBenchRubricItem(
    criterion="Important safety criterion",
    points=10,  # Higher weight
    tags=["axis:safety"]
)
```

This HealthBench-style system provides the same rigorous evaluation methodology as OpenAI's HealthBench but specifically designed for veterinary medicine and bovine disease management! 🐄
