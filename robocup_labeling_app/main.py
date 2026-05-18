import sys


def main() -> None:
    try:
        from PyQt6.QtWidgets import QApplication
        from robocup_labeling_app.viewer import LabelingWindow
    except ModuleNotFoundError as exc:
        if exc.name == "PyQt6":
            raise SystemExit(
                "PyQt6 is not installed yet. Install it with `pip install PyQt6` and run `python3 run_app.py` again."
            ) from exc
        raise

    app = QApplication(sys.argv)
    app.setApplicationName("RoboCup Labeling")
    window = LabelingWindow()
    window.show()
    sys.exit(app.exec())
