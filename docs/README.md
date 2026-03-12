# Documentation

Comprehensive documentation for the YOLO26 SeaTrash project.

## 📁 Directory Structure

```
docs/
├── guides/          # User guides and quickstart
├── reports/         # Training and integration reports
├── setup/           # Setup and configuration guides
├── status/          # Project status and summaries
├── archive/         # Historical documentation
│   ├── analysis/    # Dataset and performance analysis
│   ├── integration/ # Dataset integration history
│   ├── mapping/     # Class mapping documentation
│   └── training/    # Training history (V5-V8)
└── api/             # API documentation
```

## 📚 Key Documents

### Quick Start
- **[V11 Quickstart Guide](guides/V11_QUICKSTART.md)** - Get started with V11 training and inference

### Reports
- **[V11 Training Report](reports/V11_TRAINING_REPORT.md)** - Complete V11 training results
- **[V11 Inference Report](reports/V11_INFERENCE_REPORT.md)** - Video inference analysis
- **[Material 9-Class Integration](reports/MATERIAL_9CLASS_INTEGRATION_REPORT.md)** - Dataset integration details

### Setup
- **[Directory Structure](setup/DIRECTORY_STRUCTURE.md)** - Project organization

### Status
- **[V11 Implementation Summary](status/V11_IMPLEMENTATION_SUMMARY.md)** - Current project status

### Material Classification
- **[Material Mapping Final](MATERIAL_MAPPING_FINAL.md)** - Complete class mapping (87→9)

## 📊 V11 Performance Summary

- **mAP50**: 82.54%
- **Precision**: 85.03%
- **Recall**: 76.85%
- **Classes**: 9 material-based categories
- **Dataset**: 75,173 images, 307,530 instances
- **Training**: 272 epochs, 18h 33m (8x RTX 4090)

## 🗂️ Archive

The `archive/` directory contains historical documentation from earlier versions (V5-V8) and analysis reports. These are kept for reference but are not actively maintained.

## 📝 Document Count

- Total: 58 markdown files
- Active documentation: ~10 files
- Archived documentation: ~48 files
