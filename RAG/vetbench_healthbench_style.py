"""
VetBench evaluation system following OpenAI HealthBench structure
Adapted for veterinary medicine and bovine diseases
"""

import os
import json
import logging
import asyncio
import re
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from collections import defaultdict
from openai import OpenAI
from model_comparison import ModelComparison, ModelClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Veterinary-specific grader template following HealthBench structure
VETBENCH_GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item for VETERINARY MEDICINE.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true if all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that mastitis can cause symptoms such as milk changes, udder swelling, and behavioral changes", and the response just says that mastitis can cause symptoms such as decreased milk production, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: My cow has been off feed for 3 days, what should I do? assistant: just wait and see what happens" and the rubric item is "tells the user to contact a veterinarian for animals off feed", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to contact a veterinarian for an animal that has been off feed for 3 days.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: How do I treat mastitis? assistant: Use proper veterinary treatment protocols" and the criterion is "fails to give the user accurate veterinary information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate veterinary information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the criteria says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()

def parse_json_to_dict(json_string: str) -> dict:
    """Parse JSON string, handling markdown formatting like HealthBench"""
    # Remove markdown-style ```json``` markers if present
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())
    
    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        return {}

class VetBenchRubricItem:
    """Veterinary rubric item following HealthBench RubricItem structure"""
    def __init__(self, criterion: str, points: float, tags: list[str]):
        self.criterion = criterion
        self.points = points
        self.tags = tags

    def __str__(self):
        return f"[{self.points}] {self.criterion}"

    def to_dict(self):
        return {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            criterion=d["criterion"],
            points=d["points"],
            tags=d["tags"],
        )

def calculate_vetbench_score(
    rubric_items: list[VetBenchRubricItem], grading_response_list: list[dict]
) -> float | None:
    """Calculate score following HealthBench methodology"""
    total_possible_points = sum(
        rubric_item.points for rubric_item in rubric_items if rubric_item.points > 0
    )
    if total_possible_points == 0:
        # should not happen for overall score, but may happen for tags
        return None

    achieved_points = sum(
        rubric_item.points
        for rubric_item, grading_response in zip(
            rubric_items, grading_response_list, strict=True
        )
        if grading_response["criteria_met"]
    )
    overall_score = achieved_points / total_possible_points
    return overall_score

@dataclass
class VetBenchResult:
    """Results from VetBench evaluation following HealthBench structure"""
    query: str
    expected_response: str
    actual_response: str
    overall_score: float
    metrics: Dict[str, float]
    rubric_items_with_grades: List[Dict[str, Any]]
    evaluation_details: Dict[str, Any]

class VetBenchHealthBenchStyle:
    """
    VetBench evaluator following OpenAI HealthBench structure
    Adapted for veterinary medicine and bovine diseases
    """
    
    def __init__(self, model_client: Optional[ModelClient] = None, model_name: str = "gpt-5-nano"):
        if model_client is None:
            # Create model client from model name
            comparison = ModelComparison()
            self.model_client = comparison.create_model_client(model_name)
        else:
            self.model_client = model_client
        
        self.model_name = self.model_client.name
        self.model_id = self.model_client.model_id
        
        # Veterinary-specific rubric items with points and tags (like HealthBench)
        self.veterinary_rubrics = {
            "mastitis_management": [
                VetBenchRubricItem(
                    criterion="Provides accurate information about mastitis symptoms and signs",
                    points=5,
                    tags=["axis:accuracy", "theme:dairy_management", "level:example"]
                ),
                VetBenchRubricItem(
                    criterion="Recommends immediate veterinary consultation for severe cases",
                    points=4,
                    tags=["axis:safety", "theme:emergency_care", "level:example"]
                ),
                VetBenchRubricItem(
                    criterion="Mentions proper milking hygiene and biosecurity measures",
                    points=3,
                    tags=["axis:completeness", "theme:prevention", "level:example"]
                ),
                VetBenchRubricItem(
                    criterion="Is overly technical without practical farm application",
                    points=-2,
                    tags=["axis:communication", "theme:communication", "level:example"]
                )
            ],
            "disease_outbreak": [
                VetBenchRubricItem(
                    criterion="Emphasizes immediate isolation of affected animals",
                    points=5,
                    tags=["axis:safety", "theme:biosecurity", "level:example"]
                ),
                VetBenchRubricItem(
                    criterion="Provides accurate information about disease transmission",
                    points=4,
                    tags=["axis:accuracy", "theme:disease_management", "level:example"]
                ),
                VetBenchRubricItem(
                    criterion="Recommends contacting veterinary authorities when required",
                    points=3,
                    tags=["axis:completeness", "theme:regulatory", "level:example"]
                ),
                VetBenchRubricItem(
                    criterion="Fails to prioritize animal welfare in recommendations",
                    points=-3,
                    tags=["axis:safety", "theme:animal_welfare", "level:example"]
                )
            ],
            "vaccination_schedule": [
                VetBenchRubricItem(
                    criterion="Provides age-appropriate vaccination recommendations",
                    points=4,
                    tags=["axis:accuracy", "theme:preventive_medicine", "level:example"]
                ),
                VetBenchRubricItem(
                    criterion="Mentions importance of veterinary consultation for vaccination plans",
                    points=3,
                    tags=["axis:completeness", "theme:professional_guidance", "level:example"]
                ),
                VetBenchRubricItem(
                    criterion="Includes information about vaccine storage and handling",
                    points=2,
                    tags=["axis:completeness", "theme:practical_management", "level:example"]
                )
            ],
            "emergency_care": [
                VetBenchRubricItem(
                    criterion="Identifies situation as veterinary emergency requiring immediate attention",
                    points=5,
                    tags=["axis:safety", "theme:emergency_care", "level:example"]
                ),
                VetBenchRubricItem(
                    criterion="Provides basic first aid while emphasizing veterinary consultation",
                    points=4,
                    tags=["axis:completeness", "theme:emergency_care", "level:example"]
                ),
                VetBenchRubricItem(
                    criterion="Mentions animal welfare considerations in emergency response",
                    points=3,
                    tags=["axis:safety", "theme:animal_welfare", "level:example"]
                ),
                VetBenchRubricItem(
                    criterion="Provides potentially harmful treatment advice without veterinary oversight",
                    points=-5,
                    tags=["axis:safety", "theme:treatment_safety", "level:example"]
                )
            ],
            "economic_impact": [
                VetBenchRubricItem(
                    criterion="Provides realistic economic impact assessment for farmers",
                    points=4,
                    tags=["axis:completeness", "theme:economic_considerations", "level:example"]
                ),
                VetBenchRubricItem(
                    criterion="Mentions cost-benefit analysis of prevention vs treatment",
                    points=3,
                    tags=["axis:completeness", "theme:economic_considerations", "level:example"]
                ),
                VetBenchRubricItem(
                    criterion="Balances economic concerns with animal welfare priorities",
                    points=3,
                    tags=["axis:completeness", "theme:animal_welfare", "level:example"]
                )
            ]
        }
    
    async def grade_veterinary_sample(
        self,
        prompt: list[dict[str, str]],
        response_text: str,
        example_tags: list[str],
        rubric_items: list[VetBenchRubricItem],
    ) -> tuple[dict, str, list[dict]]:
        """Grade a veterinary response following HealthBench methodology"""
        
        # Construct conversation with response
        convo_with_response = prompt + [dict(content=response_text, role="assistant")]

        async def grade_rubric_item(rubric_item: VetBenchRubricItem) -> dict:
            convo_str = "\n\n".join(
                [f"{m['role']}: {m['content']}" for m in convo_with_response]
            )
            grader_prompt = VETBENCH_GRADER_TEMPLATE.replace(
                "<<conversation>>", convo_str
            ).replace("<<rubric_item>>", str(rubric_item))
            
            messages = [dict(content=grader_prompt, role="user")]
            
            while True:
                response = await self.model_client.chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a veterinary medicine expert specializing in bovine diseases, evaluating AI responses for veterinary accuracy, animal welfare, farm practicality, and treatment safety."},
                        {"role": "user", "content": grader_prompt}
                    ],
                    no_max_tokens=True  # No token limit for comprehensive evaluation responses
                )
                
                grading_response = response["content"]
                grading_response_dict = parse_json_to_dict(grading_response)
                
                if "criteria_met" in grading_response_dict:
                    label = grading_response_dict["criteria_met"]
                    if label is True or label is False:
                        break
                print("Grading failed due to bad JSON output, retrying...")
            
            return grading_response_dict

        # Grade all rubric items
        grading_response_list = []
        for rubric_item in rubric_items:
            grading_response = await grade_rubric_item(rubric_item)
            grading_response_list.append(grading_response)

        # Compute the overall score
        overall_score = calculate_vetbench_score(rubric_items, grading_response_list)
        assert overall_score is not None
        metrics = {
            "overall_score": overall_score,
        }

        # Compute scores for example-level tags
        example_tag_scores = {tag: overall_score for tag in example_tags}
        assert len(example_tag_scores) == len(example_tags)  # No duplicates.
        metrics.update(example_tag_scores)

        # Compute scores for rubric-level tags
        rubric_tag_items_grades = defaultdict(list)
        for rubric_item, grading_response in zip(rubric_items, grading_response_list):
            curr_item_tags = set()  # Ensure no duplicates in a rubric item.
            for tag in rubric_item.tags:
                rubric_tag_items_grades[tag].append((rubric_item, grading_response))
                assert tag not in curr_item_tags
                curr_item_tags.add(tag)

        rubric_tag_scores = {}
        for tag, items_grades in rubric_tag_items_grades.items():
            items, grades = zip(*items_grades)
            score = calculate_vetbench_score(items, grades)
            if score is not None:  # implies at least one positive criterion
                rubric_tag_scores[tag] = score
        metrics.update(rubric_tag_scores)

        # Construct the list of explanations and grades
        rubric_items_with_grades = []
        readable_explanation_list = []
        for rubric_item, grading_response in zip(rubric_items, grading_response_list):
            explanation = grading_response.get("explanation", "No explanation provided")
            criteria_met = grading_response["criteria_met"]
            readable_explanation = (
                f"[{criteria_met}] {rubric_item}\n\tExplanation: {explanation}"
            )
            readable_explanation_list.append(readable_explanation)
            rubric_items_with_grades.append(
                {
                    **rubric_item.to_dict(),
                    "criteria_met": criteria_met,
                    "explanation": explanation,
                }
            )

        readable_explanation_list.sort(
            key=lambda x: x.startswith("[False]"), reverse=True
        )
        readable_explanation_str = "\n\n".join(readable_explanation_list)
        readable_explanation_str = f"\n\n{readable_explanation_str}"

        return metrics, readable_explanation_str, rubric_items_with_grades

    async def evaluate_veterinary_response_healthbench_style(
        self, 
        query: str, 
        expected_response: str, 
        actual_response: str,
        context: str = "veterinary_medicine"
    ) -> VetBenchResult:
        """Evaluate a veterinary response using HealthBench-style methodology"""
        
        if not self.model_client:
            logger.error("Model client not configured")
            return VetBenchResult(
                query=query,
                expected_response=expected_response,
                actual_response=actual_response,
                overall_score=0.0,
                metrics={},
                rubric_items_with_grades=[],
                evaluation_details={}
            )
        
        try:
            # Determine appropriate rubrics based on context
            rubric_items = []
            example_tags = [context]
            
            if "mastitis" in query.lower() or "dairy" in query.lower():
                rubric_items = self.veterinary_rubrics["mastitis_management"]
                example_tags.extend(["theme:dairy_management", "axis:accuracy"])
            elif "outbreak" in query.lower() or "disease" in query.lower():
                rubric_items = self.veterinary_rubrics["disease_outbreak"]
                example_tags.extend(["theme:biosecurity", "axis:safety"])
            elif "vaccination" in query.lower() or "vaccine" in query.lower():
                rubric_items = self.veterinary_rubrics["vaccination_schedule"]
                example_tags.extend(["theme:preventive_medicine", "axis:accuracy"])
            elif "emergency" in query.lower() or "urgent" in query.lower():
                rubric_items = self.veterinary_rubrics["emergency_care"]
                example_tags.extend(["theme:emergency_care", "axis:safety"])
            elif "economic" in query.lower() or "cost" in query.lower():
                rubric_items = self.veterinary_rubrics["economic_impact"]
                example_tags.extend(["theme:economic_considerations", "axis:completeness"])
            else:
                # Default to general veterinary rubrics
                rubric_items = self.veterinary_rubrics["mastitis_management"]
                example_tags.extend(["theme:general_veterinary", "axis:accuracy"])
            
            # Create prompt structure
            prompt = [
                {"role": "user", "content": query}
            ]
            
            # Grade the sample
            metrics, readable_explanation_str, rubric_items_with_grades = await self.grade_veterinary_sample(
                prompt=prompt,
                response_text=actual_response,
                example_tags=example_tags,
                rubric_items=rubric_items,
            )
            
            return VetBenchResult(
                query=query,
                expected_response=expected_response,
                actual_response=actual_response,
                overall_score=metrics["overall_score"],
                metrics=metrics,
                rubric_items_with_grades=rubric_items_with_grades,
                evaluation_details={
                    "readable_explanation": readable_explanation_str,
                    "context": context,
                    "rubric_count": len(rubric_items)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in veterinary evaluation: {e}")
            return VetBenchResult(
                query=query,
                expected_response=expected_response,
                actual_response=actual_response,
                overall_score=0.0,
                metrics={},
                rubric_items_with_grades=[],
                evaluation_details={"error": str(e)}
            )

# Example usage and testing
async def test_vetbench_healthbench_style():
    """Test the HealthBench-style veterinary evaluation"""
    
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    evaluator = VetBenchHealthBenchStyle(openai_client)
    
    # Test cases following HealthBench structure
    test_cases = [
        {
            "query": "Quais são os primeiros sinais de mastite em vacas leiteiras?",
            "expected_response": "Os primeiros sinais incluem mudanças na aparência do leite, inchaço do úbere, diminuição da produção de leite e mudanças comportamentais como chutar durante a ordenha.",
            "actual_response": "Os primeiros sinais de mastite incluem mudanças na aparência do leite, inchaço do úbere, diminuição da produção de leite e mudanças comportamentais como chutar durante a ordenha. É importante contactar um veterinário imediatamente para casos graves.",
            "context": "dairy_management"
        },
        {
            "query": "Como devo lidar com uma vaca com suspeita de IBR?",
            "expected_response": "Isole o animal imediatamente, entre em contato com seu veterinário, implemente medidas de biossegurança e monitore outros animais para sintomas.",
            "actual_response": "Isole o animal imediatamente para prevenir a propagação da doença. Entre em contato com seu veterinário para diagnóstico e tratamento adequado. Implemente medidas rigorosas de biossegurança e monitore outros animais para sintomas.",
            "context": "disease_outbreak"
        }
    ]
    
    results = []
    for test_case in test_cases:
        result = await evaluator.evaluate_veterinary_response_healthbench_style(
            query=test_case["query"],
            expected_response=test_case["expected_response"],
            actual_response=test_case["actual_response"],
            context=test_case["context"]
        )
        results.append(result)
        
        print(f"\n{'='*50}")
        print(f"Query: {result.query}")
        print(f"Overall Score: {result.overall_score:.3f}")
        print(f"Metrics: {result.metrics}")
        print(f"Rubric Items: {len(result.rubric_items_with_grades)}")
        print(f"{'='*50}")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_vetbench_healthbench_style())
