"""Evaluate the frozen augmented models only after development training completes."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import analyze_semantic_fallback as report
from scripts.semantic_fallback_augmentation import OUT,PARENT,augmented_data

if __name__=='__main__':
    assert (OUT/'training_complete.json').exists() and (PARENT/'qa_complete.json').exists()
    report.OUT=OUT;report.data=augmented_data;report.EXTRA_ACTION_DIRS=[OUT/'qa'];report.main()
