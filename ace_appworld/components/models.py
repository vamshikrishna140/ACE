from dataclasses import dataclass, field
from typing import List, Optional, Dict

"""
All data models for the ACE AppWorld system.
Contains dataclasses representing episodes, steps, reflections, and curation operations.
"""

@dataclass
class Step:
    """Single step in ReAct execution"""
    step_num: int
    thought: str
    action: str
    observation: str
    success: bool = True


@dataclass
class Episode:
    """Complete episode of task execution"""
    task_id: str
    instruction: str
    steps: List[Step] = field(default_factory=list)
    final_answer: Optional[str] = None
    success: bool = False
    error: Optional[str] = None


@dataclass
class ReflectionResult:
    """Result of reflection analysis"""
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    bullet_tags: Optional[List[Dict[str, str]]] = None


@dataclass
class CurationOperation:
    """Single curation operation on playbook"""
    type: str  # "ADD", "UPDATE", "DELETE"
    section: str
    bullet_id: Optional[str] = None
    content: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class CurationResult:
    """Result of curation process"""
    reasoning: str
    operations: List[CurationOperation]