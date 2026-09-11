"""
Data pipeline package for vehicle data collection, completeness auditing, and export.
"""

def __getattr__(name: str):
    if name == "PipelineRunner":
        from data_pipeline.pipeline_runner import PipelineRunner
        return PipelineRunner
    if name == "CategoryCompleteness":
        from data_pipeline.completeness import CategoryCompleteness
        return CategoryCompleteness
    if name in ("CSVExporter", "CSV_COLUMNS"):
        from data_pipeline.export.csv_exporter import CSV_COLUMNS, CSVExporter
        return CSV_COLUMNS if name == "CSV_COLUMNS" else CSVExporter
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "CategoryCompleteness",
    "CSVExporter",
    "CSV_COLUMNS",
    "PipelineRunner",
]
