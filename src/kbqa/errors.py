"""异常层级：``code`` 给机器，``http_status`` 给 HTTP 层，``retryable`` 给重试策略。

三个字段缺一不可：

* 只有 message 的话，调用方只能靠字符串匹配判断错误类型；
* 只有 code 的话，HTTP 层要写一张 code → status 的映射表，两张表迟早对不上；
* ``retryable`` 单独标出来，是因为「重试」在检索系统里是个危险动作——
  检索超时值得重试，权限拒绝重试一万次也还是拒绝。
"""

from __future__ import annotations

from typing import Any


class KBQAError(Exception):
    """所有业务异常的基类。"""

    code = "internal_error"
    http_status = 500
    retryable = False

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
                "retryable": self.retryable,
            }
        }


class ConfigError(KBQAError):
    code = "config_error"


class DocumentFormatError(KBQAError):
    code = "document_format_error"
    http_status = 415


class IndexNotReadyError(KBQAError):
    code = "index_not_ready"
    http_status = 503
    retryable = True


class RetrievalError(KBQAError):
    code = "retrieval_error"
    http_status = 503
    retryable = True


class PermissionDeniedError(KBQAError):
    code = "permission_denied"
    http_status = 403


class EmptyQuestionError(KBQAError):
    code = "empty_question"
    http_status = 422


class GenerationError(KBQAError):
    code = "generation_error"
    http_status = 502
    retryable = True
