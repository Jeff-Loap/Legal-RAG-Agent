# -*- coding: utf-8 -*-
from __future__ import annotations

import configparser
from html import escape
from pathlib import Path
from urllib.parse import quote, unquote
import sys
from uuid import uuid4

from legal_agent import LegalRAGAgent, LegalRAGStore, get_default_config
from legal_agent.config import LLMSettings
from legal_agent.live_eval import evaluate_live_turn

import markdown
from PySide6.QtCore import QObject, QThread, QTimer, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


APP_DIR = Path(__file__).resolve().parent
CONFIG_INI_PATH = APP_DIR / "config.ini"
MARKDOWN_EXTENSIONS = ("extra", "fenced_code", "tables", "sane_lists", "nl2br")
APP_STYLE = """
QMainWindow, QWidget#appRoot {
    background: #edf6ff;
    color: #17324d;
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 14px;
}
QWidget#headerCard, QFrame#panelCard, QFrame#heroCard {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f8fcff, stop:1 #e3f0ff);
    border: 1px solid #c7ddf7;
    border-radius: 24px;
}
QFrame#subCard {
    background: rgba(255, 255, 255, 0.75);
    border: 1px solid #d6e7fb;
    border-radius: 20px;
}
QLabel#heroTitle { font-size: 28px; font-weight: 700; color: #123456; }
QLabel#heroSubtitle { font-size: 14px; color: #50708f; }
QLabel#sectionTitle { font-size: 16px; font-weight: 700; color: #19436c; }
QLabel#sectionHint { color: #6c8aa6; }
QLabel#statusChip {
    padding: 8px 14px; background: #d8ecff; color: #0e5aa8;
    border: 1px solid #b8d7fb; border-radius: 15px; font-weight: 700;
}
QLabel#sessionChip {
    padding: 8px 12px; background: #f4f9ff; border: 1px solid #d3e4f8;
    border-radius: 14px; color: #355673;
}
QLineEdit, QTextEdit, QComboBox, QListWidget, QTextBrowser {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid #c8dcf2;
    border-radius: 18px;
    padding: 10px 12px;
    selection-background-color: #b8dbff;
}
QTextEdit { padding-top: 12px; }
QPushButton {
    min-height: 42px; padding: 0 18px; border-radius: 16px; border: 1px solid #bcd4ef;
    background: #f5faff; color: #1d4368; font-weight: 600;
}
QPushButton:hover { background: #e7f2ff; }
QPushButton:disabled { background: #eff5fb; color: #90a6bc; border-color: #d6e0ea; }
QPushButton#primaryButton { background: #2d7fe5; color: white; border-color: #2d7fe5; }
QPushButton#primaryButton:hover { background: #186ed8; }
QPushButton#accentButton { background: #dff0ff; color: #0e5aa8; border-color: #b8d7fb; }
QToolButton#settingsButton {
    min-width: 108px;
    min-height: 38px;
    border-radius: 14px;
    border: 1px solid #b8d7fb;
    background: #f4faff;
    color: #0e5aa8;
    font-weight: 700;
}
QToolButton#settingsButton:hover { background: #e7f2ff; }
QListWidget { padding: 8px; }
QListWidget::item { margin: 5px 2px; padding: 10px 12px; border-radius: 14px; }
QListWidget::item:selected { background: #d9ecff; color: #0e4e90; }
QTabWidget::pane {
    border: 1px solid #d0e2f4; border-radius: 18px; background: rgba(255, 255, 255, 0.82); top: -1px;
}
QTabBar::tab {
    background: #edf5ff; border: 1px solid #d0e2f4; padding: 8px 14px;
    border-top-left-radius: 14px; border-top-right-radius: 14px; margin-right: 4px; color: #557694;
}
QTabBar::tab:selected { background: #ffffff; color: #1f4f7e; }
QProgressBar {
    border: 1px solid #c8ddf7; border-radius: 10px; background: rgba(255, 255, 255, 0.75);
    text-align: center; color: #4b6b87; min-height: 12px;
}
QProgressBar::chunk { background: #68aefc; border-radius: 10px; }
"""
CHAT_HTML_STYLE = """
<style>
body { background: transparent; color: #16324a; font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif; margin: 0; padding: 10px 8px 18px 8px; }
.message { border-radius: 24px; padding: 18px 20px; margin: 0 0 16px 0; border: 1px solid #d0e2f4; box-shadow: 0 10px 28px rgba(87, 140, 190, 0.10); }
.message.user { background: linear-gradient(135deg, #dff0ff 0%, #ecf7ff 100%); margin-left: 72px; }
.message.assistant { background: linear-gradient(135deg, #ffffff 0%, #f6fbff 100%); margin-right: 36px; }
.message.pending { border-color: #7fb8f6; box-shadow: 0 14px 30px rgba(54, 126, 214, 0.16); }
.meta { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 10px; color: #6483a0; font-size: 12px; }
.role { color: #124c84; font-weight: 700; font-size: 13px; letter-spacing: 0.4px; }
.state { background: #e0efff; color: #0f5da8; padding: 4px 10px; border-radius: 999px; font-weight: 700; }
.content { line-height: 1.72; font-size: 14px; }
.content p:first-child { margin-top: 0; }
.content p:last-child { margin-bottom: 0; }
.content code { background: #eaf3ff; color: #184f86; padding: 2px 6px; border-radius: 8px; }
.content pre { background: #0f2136; color: #edf6ff; padding: 14px 16px; border-radius: 16px; overflow-x: auto; }
.content pre code { background: transparent; color: inherit; padding: 0; }
.content blockquote { margin: 14px 0; padding: 10px 14px; border-left: 4px solid #7fb8f6; background: #f2f8ff; border-radius: 14px; }
.content table { border-collapse: collapse; width: 100%; margin: 12px 0; }
.content th, .content td { border: 1px solid #d6e7fb; padding: 8px 10px; }
.content th { background: #eff6ff; }
</style>
"""
PANEL_HTML_STYLE = """
<style>
body { background: transparent; color: #17324d; font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif; margin: 0; padding: 8px 6px 18px 6px; }
.panel-card { background: linear-gradient(135deg, #ffffff 0%, #f6fbff 100%); border: 1px solid #d0e2f4; border-radius: 22px; padding: 18px 18px; margin-bottom: 14px; }
.panel-title { font-size: 14px; font-weight: 700; color: #174977; margin-bottom: 6px; }
.panel-subtitle { color: #6b88a3; font-size: 12px; margin-bottom: 12px; }
.source-path { color: #5d7891; font-size: 12px; word-break: break-all; }
.source-action { margin: 8px 0 10px 0; }
.source-action a { display: inline-block; color: #0f5da8; text-decoration: none; font-weight: 700; padding: 6px 12px; border-radius: 999px; background: #e5f1ff; border: 1px solid #b8d7fb; }
.source-action a:hover { background: #d8ecff; }
.snippet { margin-top: 8px; line-height: 1.68; }
.panel-content code { background: #eaf3ff; color: #184f86; padding: 2px 6px; border-radius: 8px; }
.panel-content pre { background: #0f2136; color: #edf6ff; padding: 14px 16px; border-radius: 16px; overflow-x: auto; }
.panel-content table { border-collapse: collapse; width: 100%; margin-top: 10px; }
.panel-content th, .panel-content td { border: 1px solid #d6e7fb; padding: 8px 10px; }
.panel-content th { background: #eff6ff; }
.kv { display: grid; grid-template-columns: 116px 1fr; row-gap: 10px; column-gap: 12px; }
.kv-key { color: #6a87a3; font-weight: 700; }
.kv-value { color: #1c3d5e; word-break: break-word; }
</style>
"""


def load_llm_settings_from_ini() -> LLMSettings:
    parser = configparser.ConfigParser()
    if CONFIG_INI_PATH.exists():
        parser.read(CONFIG_INI_PATH, encoding="utf-8")
    return LLMSettings(
        base_url=parser.get("llm", "base_url", fallback="").strip(),
        api_key=parser.get("llm", "api_key", fallback="").strip(),
        model=parser.get("llm", "model", fallback="").strip(),
        temperature=parser.getfloat("llm", "temperature", fallback=0.1),
        max_tokens=parser.getint("llm", "max_tokens", fallback=700),
        retrieval_mode=parser.get("llm", "retrieval_mode", fallback="llm_retrieval").strip() or "llm_retrieval",
        answer_profile="quality",
    )


def save_llm_settings_to_ini(settings: LLMSettings) -> None:
    parser = configparser.ConfigParser()
    if CONFIG_INI_PATH.exists():
        parser.read(CONFIG_INI_PATH, encoding="utf-8")
    if "llm" not in parser:
        parser["llm"] = {}
    parser["llm"]["base_url"] = settings.base_url
    parser["llm"]["api_key"] = settings.api_key
    parser["llm"]["model"] = settings.model
    parser["llm"]["temperature"] = str(settings.temperature)
    parser["llm"]["max_tokens"] = str(settings.max_tokens)
    parser["llm"]["retrieval_mode"] = settings.retrieval_mode
    parser["llm"]["answer_profile"] = "quality"
    with CONFIG_INI_PATH.open("w", encoding="utf-8") as handle:
        parser.write(handle)


def truncate_text(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 1)] + "..."


def markdown_to_html(text: str) -> str:
    clean = str(text or "").strip()
    if not clean:
        return "<p>暂无内容。</p>"
    return markdown.markdown(clean, extensions=list(MARKDOWN_EXTENSIONS))


def build_panel_html(title: str, subtitle: str, inner_html: str) -> str:
    return (
        "<html><head>"
        f"{PANEL_HTML_STYLE}"
        "</head><body>"
        "<div class='panel-card'>"
        f"<div class='panel-title'>{escape(title)}</div>"
        f"<div class='panel-subtitle'>{escape(subtitle)}</div>"
        f"<div class='panel-content'>{inner_html}</div>"
        "</div></body></html>"
    )


def build_citation_open_url(source_path: str) -> str:
    return f"rag-citation://open?path={quote(source_path, safe='')}"


class ChatWorker(QObject):
    thinking_token = Signal(str)
    answer_token = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, agent: LegalRAGAgent, question: str, session_id: str, llm_settings: LLMSettings) -> None:
        super().__init__()
        self.agent = agent
        self.question = question
        self.session_id = session_id
        self.llm_settings = llm_settings

    @Slot()
    def run(self) -> None:
        final_result = None
        try:
            for event in self.agent.stream_ask(self.question, session_id=self.session_id, llm_settings=self.llm_settings):
                event_type = str(event.get("type", ""))
                if event_type == "thinking_token":
                    self.thinking_token.emit(str(event.get("content", "")))
                elif event_type == "token":
                    self.answer_token.emit(str(event.get("content", "")))
                elif event_type == "done":
                    final_result = event.get("result")
        except Exception as exc:
            self.failed.emit(" ".join(str(exc).split()) or exc.__class__.__name__)
            return
        if final_result is None:
            self.failed.emit("未收到完整回答结果。")
            return
        self.finished.emit(final_result)


class RebuildWorker(QObject):
    completed = Signal(object)
    failed = Signal(str)

    @Slot()
    def run(self) -> None:
        try:
            store = LegalRAGStore(get_default_config())
            stats = store.rebuild()
        except Exception as exc:
            self.failed.emit(" ".join(str(exc).split()) or exc.__class__.__name__)


class LiveEvalWorker(QObject):
    completed = Signal(object)
    failed = Signal(int, str)

    def __init__(self, store: LegalRAGStore, history_id: int) -> None:
        super().__init__()
        self.store = store
        self.history_id = history_id

    @Slot()
    def run(self) -> None:
        try:
            entry = self.store.get_history_entry(self.history_id)
            if entry is None:
                raise ValueError(f"历史记录不存在：{self.history_id}")
            payload = {
                "question": entry.question,
                "answer": entry.answer,
                "thinking": entry.thinking,
                "retrieved_chunks": entry.retrieved_chunks,
                "citations": entry.citations,
                "conversation_scope": entry.conversation_scope,
                "scope_reason": entry.scope_reason,
                "retrieval_mode": entry.retrieval_mode,
                "effective_question": entry.effective_question,
                "llm_error": entry.llm_error,
            }
            evaluation = evaluate_live_turn(payload)
            self.store.save_live_evaluation(self.history_id, evaluation)
            self.completed.emit(
                {
                    "history_id": self.history_id,
                    "evaluation": evaluation,
                }
            )
        except Exception as exc:
            message = " ".join(str(exc).split()) or exc.__class__.__name__
            try:
                self.store.save_live_evaluation(
                    self.history_id,
                    {
                        "status": "error",
                        "overall_score": 0.0,
                        "question_answer_overlap_score": 0.0,
                        "retrieval_support_score": 0.0,
                        "citation_link_score": 0.0,
                        "answer_length_score": 0.0,
                        "issue_count": 1,
                        "issues": [message],
                        "summary": f"实时评测失败：{message}",
                    },
                )
            except Exception:
                pass
            self.failed.emit(self.history_id, message)
            return
        self.completed.emit(stats)


class SettingsDialog(QDialog):
    settings_applied = Signal(object)
    rebuild_requested = Signal()

    def __init__(self, current_settings: LLMSettings, stats_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.resize(620, 540)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        header = QLabel("配置中心和知识库状态")
        header.setObjectName("heroTitle")
        header_hint = QLabel("这里统一管理模型配置、检索模式和索引维护。")
        header_hint.setObjectName("heroSubtitle")
        layout.addWidget(header)
        layout.addWidget(header_hint)

        config_card = QFrame()
        config_card.setObjectName("subCard")
        config_layout = QFormLayout(config_card)
        config_layout.setContentsMargins(16, 16, 16, 16)
        config_layout.setSpacing(10)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("大模型检索", "llm_retrieval")
        self.mode_combo.addItem("混合检索", "hybrid")
        self.base_url_input = QLineEdit()
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.model_input = QLineEdit()
        self.temperature_input = QLineEdit()
        self.max_tokens_input = QLineEdit()
        config_layout.addRow("问答模式", self.mode_combo)
        config_layout.addRow("Base URL", self.base_url_input)
        config_layout.addRow("API Key", self.api_key_input)
        config_layout.addRow("Model", self.model_input)
        config_layout.addRow("Temperature", self.temperature_input)
        config_layout.addRow("Max Tokens", self.max_tokens_input)
        layout.addWidget(config_card)

        stats_card = QFrame()
        stats_card.setObjectName("subCard")
        stats_layout = QVBoxLayout(stats_card)
        stats_layout.setContentsMargins(16, 16, 16, 16)
        stats_layout.setSpacing(8)
        stats_title = QLabel("知识库状态")
        stats_title.setObjectName("sectionTitle")
        self.stats_label = QLabel(stats_text)
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(stats_title)
        stats_layout.addWidget(self.stats_label)
        layout.addWidget(stats_card, 1)

        footer = QHBoxLayout()
        footer.setSpacing(10)
        self.save_button = QPushButton("保存配置")
        self.save_button.setObjectName("primaryButton")
        self.rebuild_button = QPushButton("重建索引")
        self.close_button = QPushButton("关闭")
        footer.addWidget(self.save_button)
        footer.addWidget(self.rebuild_button)
        footer.addStretch(1)
        footer.addWidget(self.close_button)
        layout.addLayout(footer)

        self.load_settings(current_settings)
        self.save_button.clicked.connect(self.apply_settings)
        self.rebuild_button.clicked.connect(self.rebuild_requested.emit)
        self.close_button.clicked.connect(self.reject)

    def load_settings(self, settings: LLMSettings) -> None:
        self.mode_combo.setCurrentIndex(max(self.mode_combo.findData(settings.retrieval_mode), 0))
        self.base_url_input.setText(settings.base_url)
        self.api_key_input.setText(settings.api_key)
        self.model_input.setText(settings.model)
        self.temperature_input.setText(str(settings.temperature))
        self.max_tokens_input.setText(str(settings.max_tokens))

    def update_stats(self, stats_text: str) -> None:
        self.stats_label.setText(stats_text)

    def current_llm_settings(self) -> LLMSettings:
        try:
            temperature = float(self.temperature_input.text().strip() or "0.1")
        except ValueError as exc:
            raise ValueError("Temperature 必须是数字。") from exc
        try:
            max_tokens = int(self.max_tokens_input.text().strip() or "700")
        except ValueError as exc:
            raise ValueError("Max Tokens 必须是整数。") from exc
        return LLMSettings(
            base_url=self.base_url_input.text().strip(),
            api_key=self.api_key_input.text().strip(),
            model=self.model_input.text().strip(),
            temperature=temperature,
            max_tokens=max_tokens,
            retrieval_mode=str(self.mode_combo.currentData()),
            answer_profile="quality",
        )

    @Slot()
    def apply_settings(self) -> None:
        try:
            settings = self.current_llm_settings()
        except ValueError as exc:
            QMessageBox.warning(self, "配置错误", str(exc))
            return
        self.settings_applied.emit(settings)
        if settings.disabled_reason:
            QMessageBox.warning(self, "配置已保存", settings.disabled_reason)
        else:
            QMessageBox.information(self, "完成", "配置已保存。")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("法律 RAG 桌面助手")
        self.resize(1480, 920)
        self.setMinimumSize(1180, 760)

        self.config = get_default_config()
        self.store = LegalRAGStore(self.config)
        self.agent = LegalRAGAgent(store=self.store, config=self.config)
        self.llm_settings = load_llm_settings_from_ini()
        self.current_session_id = uuid4().hex
        self.current_history_id = 0
        self.current_thinking_buffer = ""
        self.current_answer_buffer = ""
        self.current_citations: list[dict] = []
        self.chat_messages: list[dict[str, object]] = []
        self.pending_question = ""
        self.settings_dialog: SettingsDialog | None = None
        self.chat_thread: QThread | None = None
        self.chat_worker: ChatWorker | None = None
        self.rebuild_thread: QThread | None = None
        self.rebuild_worker: RebuildWorker | None = None
        self.live_eval_threads: dict[int, QThread] = {}
        self.live_eval_workers: dict[int, LiveEvalWorker] = {}
        self.running_label_base = "待命"
        self.running_step = 0

        self.activity_timer = QTimer(self)
        self.activity_timer.setInterval(280)
        self.activity_timer.timeout.connect(self.advance_activity_animation)

        self._build_ui()
        self.apply_global_style()
        self.refresh_store_stats()
        self.refresh_history_list()
        self.update_session_label()
        self.render_chat_history()
        self.render_thinking_panel("")
        self.render_citations_panel([])
        self.render_session_info({})
        QTimer.singleShot(1200, self.backfill_pending_live_evaluations)

    def _build_ui(self) -> None:
        root = QWidget(self)
        root.setObjectName("appRoot")
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        header_card = QWidget()
        header_card.setObjectName("headerCard")
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(24, 20, 24, 20)
        header_layout.setSpacing(18)

        title_block = QVBoxLayout()
        self.hero_title = QLabel("法律 RAG 桌面助手")
        self.hero_title.setObjectName("heroTitle")
        self.hero_subtitle = QLabel("圆角卡片式主问答窗口，支持 Markdown 渲染、历史会话切换、思考链路和引用证据展示。")
        self.hero_subtitle.setObjectName("heroSubtitle")
        title_block.addWidget(self.hero_title)
        title_block.addWidget(self.hero_subtitle)
        header_layout.addLayout(title_block, 1)

        header_side = QVBoxLayout()
        header_side.setSpacing(8)
        top_action_row = QHBoxLayout()
        top_action_row.setSpacing(8)
        self.new_session_button = QPushButton("新建会话")
        self.new_session_button.setObjectName("accentButton")
        self.settings_button = QToolButton()
        self.settings_button.setObjectName("settingsButton")
        self.settings_button.setText("⚙ 设置")
        top_action_row.addWidget(self.new_session_button)
        top_action_row.addWidget(self.settings_button)
        header_side.addLayout(top_action_row)

        self.status_chip = QLabel("待命")
        self.status_chip.setObjectName("statusChip")
        self.session_chip = QLabel("")
        self.session_chip.setObjectName("sessionChip")
        header_side.addWidget(self.status_chip, 0, Qt.AlignRight)
        header_side.addWidget(self.session_chip, 0, Qt.AlignRight)
        header_layout.addLayout(header_side)
        root_layout.addWidget(header_card)

        self.activity_bar = QProgressBar()
        self.activity_bar.setRange(0, 100)
        self.activity_bar.setValue(0)
        self.activity_bar.setTextVisible(False)
        self.activity_bar.hide()
        root_layout.addWidget(self.activity_bar)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter, 1)

        left_panel = QFrame()
        left_panel.setObjectName("panelCard")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(18, 18, 18, 18)
        left_layout.setSpacing(14)
        splitter.addWidget(left_panel)

        left_layout.addWidget(self.build_section_header("历史会话", "点击可恢复完整上下文，支持关键词筛选。"))
        history_card = self.build_sub_card()
        history_layout = QVBoxLayout(history_card)
        history_layout.setContentsMargins(16, 16, 16, 16)
        history_layout.setSpacing(10)
        self.history_search_input = QLineEdit()
        self.history_search_input.setPlaceholderText("按问题或答案关键词筛选")
        self.history_list = QListWidget()
        history_layout.addWidget(self.history_search_input)
        history_layout.addWidget(self.history_list, 1)
        left_layout.addWidget(history_card, 1)

        center_panel = QFrame()
        center_panel.setObjectName("heroCard")
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(18, 18, 18, 18)
        center_layout.setSpacing(14)
        splitter.addWidget(center_panel)

        center_layout.addWidget(self.build_section_header("主对话窗口", "支持 Markdown 渲染、长答案排版和运行态动画。"))
        chat_card = self.build_sub_card()
        chat_layout = QVBoxLayout(chat_card)
        chat_layout.setContentsMargins(10, 10, 10, 10)
        self.chat_output = QTextBrowser()
        self.chat_output.setOpenExternalLinks(True)
        self.chat_output.setReadOnly(True)
        self.chat_output_autoscroll_enabled = False
        self.chat_output_autoscroll_paused = False
        self.chat_output_scroll_syncing = False
        self.chat_output.verticalScrollBar().valueChanged.connect(self.on_chat_output_scroll_changed)
        chat_layout.addWidget(self.chat_output, 1)
        center_layout.addWidget(chat_card, 1)

        input_card = self.build_sub_card()
        input_layout = QVBoxLayout(input_card)
        input_layout.setContentsMargins(16, 14, 16, 14)
        input_layout.setSpacing(8)
        input_title = QLabel("输入问题")
        input_title.setObjectName("sectionTitle")
        input_hint = QLabel("回答会自动保存到历史会话。支持长事实、争议焦点和诉讼请求场景。")
        input_hint.setObjectName("sectionHint")
        self.question_input = QTextEdit()
        self.question_input.setPlaceholderText("请输入法律问题，可直接粘贴完整案情、争议焦点、诉状草稿或合同比对要求。")
        self.question_input.setFixedHeight(92)
        input_button_row = QHBoxLayout()
        input_button_row.setSpacing(10)
        self.send_button = QPushButton("发送问题")
        self.send_button.setObjectName("primaryButton")
        self.status_label = QLabel("待命")
        self.status_label.setObjectName("sectionHint")
        input_button_row.addWidget(self.send_button)
        input_button_row.addStretch(1)
        input_button_row.addWidget(self.status_label)
        input_layout.addWidget(input_title)
        input_layout.addWidget(input_hint)
        input_layout.addWidget(self.question_input)
        input_layout.addLayout(input_button_row)
        center_layout.addWidget(input_card)

        right_panel = QFrame()
        right_panel.setObjectName("panelCard")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(18, 18, 18, 18)
        right_layout.setSpacing(14)
        splitter.addWidget(right_panel)

        right_layout.addWidget(self.build_section_header("分析面板", "思考过程、引用证据和会话信息都在这里。"))
        detail_card = self.build_sub_card()
        detail_layout = QVBoxLayout(detail_card)
        detail_layout.setContentsMargins(10, 10, 10, 10)
        self.detail_tabs = QTabWidget()
        self.thinking_output = QTextBrowser()
        self.thinking_output.setOpenExternalLinks(True)
        self.citations_output = QTextBrowser()
        self.citations_output.setOpenExternalLinks(False)
        self.citations_output.setOpenLinks(False)
        self.citations_output.anchorClicked.connect(self.open_citation_link)
        self.session_info_output = QTextBrowser()
        self.session_info_output.setOpenExternalLinks(True)
        self.detail_tabs.addTab(self.thinking_output, "思考过程")
        self.detail_tabs.addTab(self.citations_output, "引用证据")
        self.detail_tabs.addTab(self.session_info_output, "会话信息")
        detail_layout.addWidget(self.detail_tabs)
        right_layout.addWidget(detail_card, 1)

        splitter.setSizes([280, 1040, 360])

        self.new_session_button.clicked.connect(self.start_new_session)
        self.settings_button.clicked.connect(self.open_settings_dialog)
        self.history_search_input.textChanged.connect(self.refresh_history_list)
        self.history_list.itemClicked.connect(self.load_history_session)
        self.send_button.clicked.connect(self.send_question)

    def build_section_header(self, title: str, hint: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        hint_label = QLabel(hint)
        hint_label.setObjectName("sectionHint")
        hint_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(hint_label)
        return widget

    def build_sub_card(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("subCard")
        return frame

    def set_chat_autoscroll(self, enabled: bool) -> None:
        self.chat_output_autoscroll_enabled = enabled
        self.chat_output_autoscroll_paused = False

    def restore_chat_scroll_position(self, value: int) -> None:
        bar = self.chat_output.verticalScrollBar()
        target = max(0, min(value, bar.maximum()))
        self.chat_output_scroll_syncing = True
        try:
            bar.setValue(target)
        finally:
            self.chat_output_scroll_syncing = False

    def scroll_chat_output_to_bottom(self) -> None:
        bar = self.chat_output.verticalScrollBar()
        self.chat_output_scroll_syncing = True
        try:
            bar.setValue(bar.maximum())
        finally:
            self.chat_output_scroll_syncing = False

    @Slot(int)
    def on_chat_output_scroll_changed(self, value: int) -> None:
        if self.chat_output_scroll_syncing or not self.chat_output_autoscroll_enabled:
            return
        bar = self.chat_output.verticalScrollBar()
        if value >= max(0, bar.maximum() - 2):
            self.chat_output_autoscroll_paused = False
        else:
            self.chat_output_autoscroll_paused = True

    def apply_global_style(self) -> None:
        self.setStyleSheet(APP_STYLE)

    def current_llm_settings(self) -> LLMSettings:
        return self.llm_settings

    def update_session_label(self) -> None:
        self.session_chip.setText(f"Session · {self.current_session_id}")

    def build_store_stats_text(self) -> str:
        stats = self.store.get_stats()
        return "\n".join(
            [
                f"当前模式：{self.llm_settings.retrieval_mode}",
                f"文档数：{stats.documents}",
                f"Chunk 数：{stats.chunks}",
                f"SQLite：{self.config.sqlite_path}",
                f"向量索引：{self.config.faiss_path}",
            ]
        )

    def refresh_store_stats(self) -> None:
        if self.settings_dialog is not None:
            self.settings_dialog.update_stats(self.build_store_stats_text())

    def open_settings_dialog(self) -> None:
        if self.settings_dialog is None:
            self.settings_dialog = SettingsDialog(
                current_settings=self.llm_settings,
                stats_text=self.build_store_stats_text(),
                parent=self,
            )
            self.settings_dialog.settings_applied.connect(self.apply_settings_from_dialog)
            self.settings_dialog.rebuild_requested.connect(self.rebuild_index)
            self.settings_dialog.finished.connect(self.on_settings_dialog_finished)
        else:
            self.settings_dialog.load_settings(self.llm_settings)
            self.settings_dialog.update_stats(self.build_store_stats_text())
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    @Slot(object)
    def apply_settings_from_dialog(self, settings: object) -> None:
        if not isinstance(settings, LLMSettings):
            return
        self.llm_settings = settings
        save_llm_settings_to_ini(settings)
        self.refresh_store_stats()
        if settings.disabled_reason:
            self.status_chip.setText("配置已保存，但不可用")
            self.status_label.setText("配置已保存，但不可用")
        else:
            self.status_chip.setText("配置已保存")
            self.status_label.setText("配置已保存")

    @Slot(int)
    def on_settings_dialog_finished(self, _result: int) -> None:
        if self.settings_dialog is not None:
            self.settings_dialog.deleteLater()
        self.settings_dialog = None

    def refresh_history_list(self) -> None:
        keyword = self.history_search_input.text().strip()
        entries = self.store.list_history_entries(limit=120, keyword=keyword)
        latest_by_session: dict[str, object] = {}
        for entry in entries:
            if entry.session_id not in latest_by_session:
                latest_by_session[entry.session_id] = entry

        self.history_list.clear()
        for entry in latest_by_session.values():
            label = f"{entry.created_at} | {truncate_text(entry.question, 30)}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, entry.session_id)
            item.setToolTip(entry.question)
            self.history_list.addItem(item)

    def render_chat_history(self) -> None:
        previous_scroll_value = self.chat_output.verticalScrollBar().value()
        blocks: list[str] = []
        for message in self.chat_messages:
            role = str(message.get("role", "assistant"))
            content = str(message.get("content", "") or "")
            pending = bool(message.get("pending", False))
            role_label = "用户" if role == "user" else "法律助手"
            state_label = str(message.get("state", "") or "")
            if state_label:
                meta_html = (
                    "<div class='meta'>"
                    f"<span class='role'>{escape(role_label)}</span>"
                    f"<span class='state'>{escape(state_label)}</span>"
                    "</div>"
                )
            else:
                meta_html = f"<div class='meta'><span class='role'>{escape(role_label)}</span></div>"
            body_html = markdown_to_html(content if content.strip() else "思考中...")
            classes = f"message {'user' if role == 'user' else 'assistant'}"
            if pending:
                classes += " pending"
            blocks.append(f"<section class='{classes}'>{meta_html}<div class='content'>{body_html}</div></section>")

        if not blocks:
            blocks.append(
                "<section class='message assistant'>"
                "<div class='meta'><span class='role'>法律助手</span></div>"
                "<div class='content'><p>请输入问题开始对话。当前窗口支持 <code>Markdown</code> 渲染、历史会话恢复、思考过程和引用证据查看。</p></div>"
                "</section>"
            )

        html = f"<html><head>{CHAT_HTML_STYLE}</head><body>{''.join(blocks)}</body></html>"
        self.chat_output_scroll_syncing = True
        try:
            self.chat_output.setHtml(html)
        finally:
            self.chat_output_scroll_syncing = False

        if self.chat_output_autoscroll_enabled and not self.chat_output_autoscroll_paused:
            self.scroll_chat_output_to_bottom()
        else:
            self.restore_chat_scroll_position(previous_scroll_value)

    def render_thinking_panel(self, thinking: str) -> None:
        subtitle = "完整展示当前轮的思考摘要、检索语句和阶段信息。"
        inner_html = markdown_to_html(thinking or "暂无思考过程。")
        self.thinking_output.setHtml(build_panel_html("思考过程", subtitle, inner_html))

    def render_citations_panel(self, citations: list[dict]) -> None:
        if not citations:
            self.citations_output.setHtml(build_panel_html("引用证据", "本轮尚未生成引用。", "<p>暂无引用证据。</p>"))
            return

        cards: list[str] = []
        for index, citation in enumerate(citations, start=1):
            label = escape(str(citation.get("label", f"引用 {index}")))
            source_path = str(citation.get("source_path", "") or "").strip()
            source_name = escape(str(citation.get("source_name", "") or "未知来源"))
            title = escape(str(citation.get("title", "") or ""))
            page_start = citation.get("page_start")
            page_end = citation.get("page_end")
            page_text = ""
            if page_start and page_end:
                page_text = f"第 {page_start} 页"
                if page_start != page_end:
                    page_text = f"第 {page_start}-{page_end} 页"
            open_href = build_citation_open_url(source_path) if source_path else ""
            if open_href:
                path_html = (
                    f"<div class='source-path'>{source_name}"
                    + (f" · {title}" if title else "")
                    + (f" · {escape(page_text)}" if page_text else "")
                    + "</div>"
                    f"<div class='source-action'><a href='{escape(open_href)}'>打开原文</a></div>"
                    f"<div class='source-path'>{escape(source_path)}</div>"
                )
            else:
                path_html = f"<div class='source-path'>{source_name}</div><div class='source-path'>{escape(source_path) or '未记录路径'}</div>"
            snippet_html = markdown_to_html(str(citation.get("snippet", "") or ""))
            cards.append(
                "<div class='panel-card'>"
                f"<div class='panel-title'>{label}</div>"
                "<div class='panel-subtitle'>来自本地知识库的命中证据。</div>"
                f"{path_html}"
                f"<div class='snippet'>{snippet_html}</div>"
                "</div>"
            )

        html = f"<html><head>{PANEL_HTML_STYLE}</head><body>{''.join(cards)}</body></html>"
        self.citations_output.setHtml(html)

    @Slot(QUrl)
    def open_citation_link(self, url: QUrl) -> None:
        if not url.isValid():
            QMessageBox.warning(self, "打开失败", "引用链接无效。")
            return

        if url.scheme() == "rag-citation":
            raw_path = url.query().split("path=", 1)[-1] if "path=" in url.query() else ""
            if not raw_path:
                QMessageBox.warning(self, "打开失败", "引用链接缺少文件路径。")
                return
            source_path = unquote(raw_path)
            local_file = Path(str(source_path))
            if not local_file.exists():
                QMessageBox.warning(self, "打开失败", f"文件不存在：{local_file}")
                return
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(local_file))):
                QMessageBox.warning(self, "打开失败", f"无法打开文件：{local_file}")
            return

        if url.isLocalFile():
            local_file = Path(url.toLocalFile())
            if not local_file.exists():
                QMessageBox.warning(self, "打开失败", f"文件不存在：{local_file}")
                return
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(local_file))):
                QMessageBox.warning(self, "打开失败", f"无法打开文件：{local_file}")
            return

        if not QDesktopServices.openUrl(url):
            QMessageBox.warning(self, "打开失败", f"无法打开链接：{url.toString()}")

    def render_session_info(self, payload: dict[str, object]) -> None:
        if not payload:
            self.session_info_output.setHtml(
                build_panel_html("会话信息", "当前还没有会话指标。", "<p>发送问题后会在这里展示检索模式、命中数量和时间信息。</p>")
            )
            return

        rows: list[str] = []
        for key, value in payload.items():
            rows.append(
                f"<div class='kv-key'>{escape(str(key))}</div>"
                f"<div class='kv-value'>{escape(str(value))}</div>"
            )
        inner_html = f"<div class='kv'>{''.join(rows)}</div>"
        self.session_info_output.setHtml(build_panel_html("会话信息", "当前轮运行状态与结果元数据。", inner_html))

    def set_busy_state(self, active: bool, base_label: str) -> None:
        if active:
            self.running_label_base = base_label
            self.running_step = 0
            self.activity_bar.setRange(0, 0)
            self.activity_bar.show()
            self.activity_timer.start()
            self.advance_activity_animation()
            return

        self.activity_timer.stop()
        self.activity_bar.hide()
        self.activity_bar.setRange(0, 100)
        self.activity_bar.setValue(0)
        self.status_chip.setText(base_label)
        self.status_label.setText(base_label)

    def advance_activity_animation(self) -> None:
        suffix = "." * ((self.running_step % 3) + 1)
        animated = f"{self.running_label_base}{suffix}"
        self.status_chip.setText(animated)
        self.status_label.setText(animated)
        if self.chat_messages and bool(self.chat_messages[-1].get("pending", False)):
            self.chat_messages[-1]["state"] = animated
            self.render_chat_history()
        self.running_step += 1

    def start_new_session(self) -> None:
        if self.chat_thread is not None:
            QMessageBox.warning(self, "运行中", "当前仍在生成回答，不能切换会话。")
            return
        self.set_chat_autoscroll(False)
        self.current_session_id = uuid4().hex
        self.current_history_id = 0
        self.update_session_label()
        self.chat_messages = []
        self.current_thinking_buffer = ""
        self.current_answer_buffer = ""
        self.current_citations = []
        self.render_chat_history()
        self.render_thinking_panel("")
        self.render_citations_panel([])
        self.render_session_info({})
        self.status_chip.setText("新会话已创建")
        self.status_label.setText("新会话已创建")

    def save_current_config(self) -> None:
        self.open_settings_dialog()

    def rebuild_index(self) -> None:
        if self.chat_thread is not None or self.rebuild_thread is not None:
            QMessageBox.warning(self, "运行中", "当前有任务正在执行，请稍后。")
            return

        self.rebuild_thread = QThread(self)
        self.rebuild_worker = RebuildWorker()
        self.rebuild_worker.moveToThread(self.rebuild_thread)
        self.rebuild_thread.started.connect(self.rebuild_worker.run)
        self.rebuild_worker.completed.connect(self.on_rebuild_completed)
        self.rebuild_worker.failed.connect(self.on_rebuild_failed)
        self.rebuild_thread.finished.connect(self.cleanup_rebuild_thread)
        self.send_button.setEnabled(False)
        self.set_busy_state(True, "索引重建中")
        self.rebuild_thread.start()

    @Slot(object)
    def on_rebuild_completed(self, _stats: object) -> None:
        self.agent.refresh()
        self.refresh_store_stats()
        self.set_busy_state(False, "索引重建完成")
        if self.rebuild_thread is not None:
            self.rebuild_thread.quit()
        QMessageBox.information(self, "完成", "知识库索引已重建。")

    @Slot(str)
    def on_rebuild_failed(self, message: str) -> None:
        self.set_busy_state(False, "索引重建失败")
        if self.rebuild_thread is not None:
            self.rebuild_thread.quit()
        QMessageBox.warning(self, "失败", message)

    @Slot()
    def cleanup_rebuild_thread(self) -> None:
        self.send_button.setEnabled(True)
        if self.rebuild_worker is not None:
            self.rebuild_worker.deleteLater()
        if self.rebuild_thread is not None:
            self.rebuild_thread.deleteLater()
        self.rebuild_worker = None
        self.rebuild_thread = None

    def send_question(self) -> None:
        if self.chat_thread is not None:
            QMessageBox.warning(self, "运行中", "当前仍在生成回答，请稍后。")
            return

        question = self.question_input.toPlainText().strip()
        if not question:
            QMessageBox.warning(self, "缺少问题", "请输入问题后再发送。")
            return

        try:
            llm_settings = self.current_llm_settings()
        except ValueError as exc:
            QMessageBox.warning(self, "配置错误", str(exc))
            return
        if llm_settings.disabled_reason:
            QMessageBox.warning(self, "配置错误", llm_settings.disabled_reason)
            return

        self.set_chat_autoscroll(True)
        self.current_thinking_buffer = ""
        self.current_answer_buffer = ""
        self.current_citations = []
        self.render_thinking_panel("")
        self.render_citations_panel([])
        self.render_session_info({})

        self.chat_messages.append({"role": "user", "content": question, "state": "已提交", "pending": False})
        self.chat_messages.append({"role": "assistant", "content": "", "state": "思考中.", "pending": True})
        self.render_chat_history()

        self.question_input.clear()
        self.pending_question = question
        self.send_button.setEnabled(False)
        self.set_busy_state(True, "思考中")

        self.chat_thread = QThread(self)
        self.chat_worker = ChatWorker(agent=self.agent, question=question, session_id=self.current_session_id, llm_settings=llm_settings)
        self.chat_worker.moveToThread(self.chat_thread)
        self.chat_thread.started.connect(self.chat_worker.run)
        self.chat_worker.thinking_token.connect(self.on_thinking_token)
        self.chat_worker.answer_token.connect(self.on_answer_token)
        self.chat_worker.finished.connect(self.on_chat_finished)
        self.chat_worker.failed.connect(self.on_chat_failed)
        self.chat_thread.finished.connect(self.cleanup_chat_thread)
        self.chat_thread.start()

    @Slot(str)
    def on_thinking_token(self, token: str) -> None:
        self.current_thinking_buffer += token
        self.render_thinking_panel(self.current_thinking_buffer)

    @Slot(str)
    def on_answer_token(self, token: str) -> None:
        self.current_answer_buffer += token
        if self.chat_messages:
            self.chat_messages[-1]["content"] = self.current_answer_buffer
            self.render_chat_history()

    @Slot(object)
    def on_chat_finished(self, result: object) -> None:
        if not isinstance(result, dict):
            self.on_chat_failed("返回结果格式错误。")
            return

        final_answer = str(result.get("answer", "") or self.current_answer_buffer).strip()
        if not final_answer:
            self.on_chat_failed("返回答案为空。")
            return

        self.current_citations = list(result.get("citations", []) or [])
        llm_error = str(result.get("llm_error", "") or "").strip()
        mode_name = str(result.get("retrieval_mode", "") or self.llm_settings.retrieval_mode)
        if self.chat_messages:
            self.chat_messages[-1]["content"] = final_answer
            self.chat_messages[-1]["pending"] = False
            self.chat_messages[-1]["state"] = f"{mode_name} · 已完成"
            self.render_chat_history()
        self.set_chat_autoscroll(False)

        info_payload = {
            "session_id": self.current_session_id,
            "retrieval_mode": mode_name,
            "conversation_scope": result.get("conversation_scope", ""),
            "llm_used": bool(result.get("llm_used", False)),
            "citations": len(self.current_citations),
            "memory_hits": len(result.get("memory_hits", []) or []),
            "retrieved_chunks": len(result.get("retrieved_chunks", []) or []),
        }
        if llm_error:
            info_payload["llm_error"] = llm_error
        scope_reason = str(result.get("scope_reason", "") or "").strip()
        if scope_reason:
            info_payload["scope_reason"] = scope_reason
        info_payload["live_eval"] = "排队中"

        self.render_thinking_panel(str(result.get("thinking", "") or self.current_thinking_buffer))
        self.render_citations_panel(self.current_citations)
        self.render_session_info(info_payload)

        saved_id = self.store.save_history_entry(
            session_id=self.current_session_id,
            question=self.pending_question,
            answer=final_answer,
            thinking=str(result.get("thinking", "") or self.current_thinking_buffer),
            citations=self.current_citations,
            llm_used=bool(result.get("llm_used", False)),
            llm_error=llm_error,
            retrieved_chunks=list(result.get("retrieved_chunks", []) or []),
            conversation_scope=str(result.get("conversation_scope", "") or ""),
            scope_reason=str(result.get("scope_reason", "") or ""),
            retrieval_mode=mode_name,
            effective_question=str(result.get("effective_question", "") or ""),
        )
        saved_entry = self.store.get_history_entry(saved_id)
        if saved_entry is None:
            self.on_chat_failed(f"历史记录保存后读取失败：{saved_id}")
            return

        self.current_history_id = saved_id
        self.start_live_evaluation(saved_id)

        self.refresh_history_list()
        self.set_busy_state(False, "回答完成")
        if self.chat_thread is not None:
            self.chat_thread.quit()
        self.pending_question = ""

    @Slot(str)
    def on_chat_failed(self, message: str) -> None:
        self.set_chat_autoscroll(False)
        if self.chat_messages and self.chat_messages[-1].get("role") == "assistant":
            current = str(self.chat_messages[-1].get("content", "") or "")
            error_markdown = f"{current}\n\n> [错误] {message}".strip()
            self.chat_messages[-1]["content"] = error_markdown
            self.chat_messages[-1]["pending"] = False
            self.chat_messages[-1]["state"] = "运行失败"
            self.render_chat_history()
        self.render_session_info({"session_id": self.current_session_id, "error": message})
        self.set_busy_state(False, "回答失败")
        if self.chat_thread is not None:
            self.chat_thread.quit()
        self.pending_question = ""
        QMessageBox.warning(self, "失败", message)

    def start_live_evaluation(self, history_id: int) -> None:
        if history_id in self.live_eval_threads:
            return
        try:
            entry = self.store.get_history_entry(history_id)
            if entry is None:
                return
            self.store.save_live_evaluation(
                history_id,
                {
                    "status": "processing",
                    "overall_score": 0.0,
                    "question_answer_overlap_score": 0.0,
                    "retrieval_support_score": 0.0,
                    "citation_link_score": 0.0,
                    "answer_length_score": 0.0,
                    "issue_count": 0,
                    "issues": [],
                    "summary": "实时评测处理中...",
                },
            )
        except Exception as exc:
            QMessageBox.warning(self, "实时评测启动失败", " ".join(str(exc).split()) or exc.__class__.__name__)
            return
        thread = QThread(self)
        worker = LiveEvalWorker(self.store, history_id)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self.on_live_eval_completed)
        worker.failed.connect(self.on_live_eval_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(lambda hid=history_id: self.cleanup_live_eval_thread(hid))
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self.live_eval_threads[history_id] = thread
        self.live_eval_workers[history_id] = worker
        thread.start()

    def backfill_pending_live_evaluations(self) -> None:
        pending_entries = self.store.list_pending_history_entries(limit=50)
        for entry in pending_entries:
            self.start_live_evaluation(entry.id)
        if pending_entries:
            self.status_label.setText(f"实时评测补评 {len(pending_entries)} 条")

    @Slot(object)
    def on_live_eval_completed(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        history_id = int(payload.get("history_id", 0) or 0)
        evaluation = payload.get("evaluation", {}) or {}
        self.cleanup_live_eval_thread(history_id)

        score = float(evaluation.get("overall_score", 0.0) or 0.0)
        summary = str(evaluation.get("summary", "") or "")
        issue_count = int(evaluation.get("issue_count", 0) or 0)
        if history_id == self.current_history_id:
            stored_evaluation = self.store.get_live_evaluation_by_history_id(history_id)
            if stored_evaluation is None:
                stored_evaluation = {}
            self.render_session_info(
                {
                    "session_id": stored_evaluation.get("session_id", self.current_session_id),
                    "turn_id": stored_evaluation.get("turn_id", history_id),
                    "retrieval_mode": stored_evaluation.get("retrieval_mode", self.llm_settings.retrieval_mode),
                    "conversation_scope": stored_evaluation.get("conversation_scope", ""),
                    "live_eval_score": f"{score * 100:.1f}%",
                    "live_eval_summary": stored_evaluation.get("summary", summary),
                    "live_eval_issues": stored_evaluation.get("issue_count", issue_count),
                }
            )
            self.status_chip.setText(f"实时评测 {score * 100:.1f}%")
            self.status_label.setText(f"实时评测 {score * 100:.1f}%")
        elif summary:
            self.status_label.setText(f"实时评测 {score * 100:.1f}%")

    @Slot(int, str)
    def on_live_eval_failed(self, history_id: int, message: str) -> None:
        if history_id == self.current_history_id:
            self.status_label.setText("实时评测失败")
            self.status_chip.setText("实时评测失败")
            self.render_session_info({"session_id": self.current_session_id, "live_eval_error": message})

    def cleanup_live_eval_thread(self, history_id: int) -> None:
        self.live_eval_threads.pop(history_id, None)
        self.live_eval_workers.pop(history_id, None)

    @Slot()
    def cleanup_chat_thread(self) -> None:
        self.send_button.setEnabled(True)
        if self.chat_worker is not None:
            self.chat_worker.deleteLater()
        if self.chat_thread is not None:
            self.chat_thread.deleteLater()
        self.chat_worker = None
        self.chat_thread = None

    @Slot(QListWidgetItem)
    def load_history_session(self, item: QListWidgetItem) -> None:
        if self.chat_thread is not None:
            QMessageBox.warning(self, "运行中", "当前仍在生成回答，不能加载历史会话。")
            return
        self.set_chat_autoscroll(False)
        session_id = str(item.data(Qt.UserRole) or "").strip()
        if not session_id:
            QMessageBox.warning(self, "缺少会话", "该历史项缺少 session_id。")
            return
        entries = self.store.list_session_entries(session_id)
        if not entries:
            QMessageBox.warning(self, "空会话", "未找到该历史会话内容。")
            return

        self.current_session_id = session_id
        self.update_session_label()
        self.chat_messages = []
        for entry in entries:
            self.chat_messages.append({"role": "user", "content": entry.question, "state": entry.created_at, "pending": False})
            self.chat_messages.append({"role": "assistant", "content": entry.answer, "state": f"turn {entry.turn_id}", "pending": False})
        self.render_chat_history()

        latest_entry = entries[-1]
        self.current_history_id = latest_entry.id
        self.render_thinking_panel(latest_entry.thinking)
        citations = [item for item in latest_entry.citations if isinstance(item, dict)]
        self.render_citations_panel(citations)
        info_payload = {
            "session_id": session_id,
            "history_turns": len(entries),
            "latest_turn_id": latest_entry.turn_id,
            "latest_created_at": latest_entry.created_at,
        }
        evaluation = self.store.get_live_evaluation_by_history_id(latest_entry.id)
        if evaluation is not None:
            info_payload["live_eval_score"] = f"{float(evaluation.get('overall_score', 0.0)) * 100:.1f}%"
            info_payload["live_eval_summary"] = evaluation.get("summary", "")
            info_payload["live_eval_issues"] = int(evaluation.get("issue_count", 0) or 0)
        self.render_session_info(info_payload)
        self.status_chip.setText("已加载历史会话")
        self.status_label.setText("已加载历史会话")


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
