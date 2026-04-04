# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = APP_DIR / "eval" / "legal_qa_benchmark.json"
DEFAULT_CONFIG_INI = APP_DIR / "config.ini"
DEFAULT_OUTPUT = APP_DIR / "eval" / "reports" / "legal_rag_harness_latest.json"
MODE_SPECS = ("hybrid:quality", "llm_retrieval:quality")


class HarnessWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("法律 RAG Harness")
        self.resize(980, 720)
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(APP_DIR))
        self.process.readyReadStandardOutput.connect(self._append_stdout)
        self.process.readyReadStandardError.connect(self._append_stderr)
        self.process.finished.connect(self._handle_finished)

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        paths_box = QGroupBox("路径")
        paths_layout = QFormLayout(paths_box)
        self.benchmark_input = QLineEdit(str(DEFAULT_BENCHMARK))
        self.config_input = QLineEdit(str(DEFAULT_CONFIG_INI))
        self.output_input = QLineEdit(str(DEFAULT_OUTPUT))
        paths_layout.addRow("Benchmark", self._build_path_row(self.benchmark_input, "open_json"))
        paths_layout.addRow("Config INI", self._build_path_row(self.config_input, "ini"))
        paths_layout.addRow("Output", self._build_path_row(self.output_input, "save_json"))
        root_layout.addWidget(paths_box)

        options_box = QGroupBox("选项")
        options_layout = QGridLayout(options_box)
        self.top_k_input = QSpinBox()
        self.top_k_input.setRange(0, 100)
        self.top_k_input.setValue(0)
        self.top_k_input.setSpecialValueText("默认")
        self.details_checkbox = QCheckBox("输出详细结果")
        self.skip_checkbox = QCheckBox("跳过未配置 LLM 的模式")
        self.skip_checkbox.setChecked(True)
        options_layout.addWidget(QLabel("Top K"), 0, 0)
        options_layout.addWidget(self.top_k_input, 0, 1)
        options_layout.addWidget(self.details_checkbox, 0, 2)
        options_layout.addWidget(self.skip_checkbox, 0, 3)
        root_layout.addWidget(options_box)

        modes_box = QGroupBox("模式")
        modes_layout = QGridLayout(modes_box)
        self.mode_checkboxes: dict[str, QCheckBox] = {}
        for index, spec in enumerate(MODE_SPECS):
            checkbox = QCheckBox(spec)
            checkbox.setChecked(True)
            modes_layout.addWidget(checkbox, index // 2, index % 2)
            self.mode_checkboxes[spec] = checkbox
        root_layout.addWidget(modes_box)

        button_layout = QHBoxLayout()
        self.run_button = QPushButton("运行 Harness")
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.open_output_button = QPushButton("打开输出目录")
        self.clear_log_button = QPushButton("清空日志")
        self.status_label = QLabel("待运行")
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.open_output_button)
        button_layout.addWidget(self.clear_log_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.status_label)
        root_layout.addLayout(button_layout)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        root_layout.addWidget(self.log_output, 1)

        self.run_button.clicked.connect(self.run_harness)
        self.stop_button.clicked.connect(self.stop_harness)
        self.open_output_button.clicked.connect(self.open_output_dir)
        self.clear_log_button.clicked.connect(self.log_output.clear)

    def _build_path_row(self, line_edit: QLineEdit, suffix: str) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        browse_button = QPushButton("浏览")
        browse_button.clicked.connect(lambda: self._select_path(line_edit, suffix))
        layout.addWidget(line_edit, 1)
        layout.addWidget(browse_button)
        return container

    def _select_path(self, line_edit: QLineEdit, suffix: str) -> None:
        current = line_edit.text().strip() or str(APP_DIR)
        if suffix == "ini":
            path, _ = QFileDialog.getOpenFileName(self, "选择 INI 文件", current, "INI (*.ini);;All Files (*)")
        elif suffix == "open_json":
            path, _ = QFileDialog.getOpenFileName(self, "选择 JSON 文件", current, "JSON (*.json);;All Files (*)")
        elif suffix == "save_json":
            path, _ = QFileDialog.getSaveFileName(self, "选择 JSON 文件", current, "JSON (*.json);;All Files (*)")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择文件", current, "All Files (*)")
        if path:
            line_edit.setText(path)

    def selected_modes(self) -> list[str]:
        return [name for name, checkbox in self.mode_checkboxes.items() if checkbox.isChecked()]

    def run_harness(self) -> None:
        if self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "运行中", "Harness 正在运行，请先停止当前任务。")
            return
        modes = self.selected_modes()
        if not modes:
            QMessageBox.warning(self, "缺少模式", "至少勾选一个评测模式。")
            return

        benchmark_path = Path(self.benchmark_input.text().strip())
        if not benchmark_path.exists():
            QMessageBox.warning(self, "缺少 benchmark", f"找不到 benchmark 文件：{benchmark_path}")
            return

        command = [
            str(APP_DIR / "run_legal_rag_harness.py"),
            "--benchmark",
            str(benchmark_path),
            "--config-ini",
            self.config_input.text().strip(),
            "--output",
            self.output_input.text().strip(),
            "--modes",
            *modes,
        ]
        if self.top_k_input.value() > 0:
            command.extend(["--top-k", str(self.top_k_input.value())])
        if self.details_checkbox.isChecked():
            command.append("--details")
        if self.skip_checkbox.isChecked():
            command.append("--skip-unavailable-modes")

        self.log_output.appendPlainText(f"$ {sys.executable} {' '.join(command)}")
        self.process.start(sys.executable, command)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("运行中")

    def stop_harness(self) -> None:
        if self.process.state() == QProcess.NotRunning:
            return
        self.process.kill()
        self.process.waitForFinished(3000)
        self.status_label.setText("已停止")
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def open_output_dir(self) -> None:
        output_dir = str(Path(self.output_input.text().strip()).resolve().parent)
        QProcess.startDetached("explorer.exe", [output_dir])

    def _append_stdout(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self.log_output.appendPlainText(text.rstrip())

    def _append_stderr(self) -> None:
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        if text:
            self.log_output.appendPlainText(text.rstrip())

    def _handle_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText(f"完成，exit_code={exit_code}")
        if exit_code != 0:
            QMessageBox.warning(self, "评测失败", f"Harness 退出码非 0：{exit_code}")


def main() -> int:
    app = QApplication(sys.argv)
    window = HarnessWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
