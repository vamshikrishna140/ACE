from dataclasses import dataclass, field
from typing import List, Optional

"""
This file contains the core data structures (dataclasses) used to represent
task episodes and individual steps.
(This file is unchanged from your original `models.py`)
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

