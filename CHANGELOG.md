# Changelog

## Unreleased

- 建立 C++20/CMake 项目、公共 `nivo` API 与可安装的 `Nivo::cache` target。
- 定义 pool/key、草稿发布、只读快照、版本 CAS、TTL、容量与关闭语义。
- 实现单锁内存参考引擎，支持 LRU/FIFO、统计与 pool resize。
- 添加契约/并发测试、示例、基准入口与 sanitizer CI 检查。
- 明确 SSD、异步、持久化、chained/object cache 尚未实现。
