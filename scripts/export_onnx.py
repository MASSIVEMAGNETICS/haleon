#!/usr/bin/env python3
"""
Export Eidolon-Ω model to ONNX format for production deployment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eidolon-inference'))

import torch
from eidolon_model import TextToMusicModel, export_to_onnx


def main():
    """Export model to ONNX."""
    print("Initializing Eidolon-Ω model...")
    model = TextToMusicModel(
        dim=768,
        num_layers=12,
        num_heads=12,
        max_seq_len=512,
        vocab_size=50000,
        output_dim=128
    )
    
    # Initialize with random weights (in production, load trained weights)
    print("Model initialized with random weights")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Export to ONNX
    output_path = "eidolon.onnx"
    print(f"\nExporting to ONNX: {output_path}")
    
    export_to_onnx(
        model=model,
        output_path=output_path,
        batch_size=1,
        seq_len=128
    )
    
    print(f"✓ Successfully exported to {output_path}")
    
    # Verify the exported model
    import onnxruntime as ort
    print("\nVerifying ONNX model...")
    session = ort.InferenceSession(output_path)
    
    print("Model inputs:")
    for inp in session.get_inputs():
        print(f"  - {inp.name}: {inp.shape} ({inp.type})")
    
    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  - {out.name}: {out.shape} ({out.type})")
    
    print("\n✓ ONNX model verified successfully!")
    print(f"\nCopy {output_path} to eidolon-inference/ directory for deployment")


if __name__ == "__main__":
    main()
