"""User-facing failure categories without exposing provider response bodies."""
from urllib.error import HTTPError, URLError


def image_error_detail(error):
    cause = error
    known = {
        '恢复生图的参数已改变，请使用新的批次编号',
        '生图提交结果未知，请核查云端任务后再决定是否新建批次',
    }
    for _ in range(8):
        if isinstance(cause, HTTPError):
            return {
                401: '认证失败，请更新阿里云凭据后恢复',
                403: '访问被拒绝，请检查模型权限和账户状态',
                429: '请求受限，请稍后恢复并检查账户额度',
            }.get(cause.code, f'云端请求失败（HTTP {cause.code}），请检查服务后恢复')
        if isinstance(cause, TimeoutError):
            return '等待超时，任务可能仍在运行；请恢复原批次'
        if isinstance(cause, URLError):
            return '网络连接失败，请检查网络后恢复原批次'
        if str(cause) in known:
            return str(cause)
        cause = cause.__cause__
        if cause is None:
            break
    return '生成未完成，请检查服务配置并恢复原批次'
