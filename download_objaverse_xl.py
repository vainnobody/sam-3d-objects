#!/usr/bin/env python3
"""
Objaverse-XL 数据集下载脚本

Objaverse-XL 是一个包含 10M+ 3D 对象的大型数据集。
数据来源: GitHub, Thingiverse, Sketchfab, Smithsonian

功能:
- 下载全部标注信息
- 批量下载 3D 对象
- 支持断点续传
- 进度跟踪和日志记录
- 错误处理和重试机制
- 支持镜像站下载 (国内推荐)

使用方法:
    # 完整下载
    python download_objaverse_xl.py

    # 测试下载 (100个对象)
    python download_objaverse_xl.py --sample_size 100

    # 使用 HuggingFace 镜像 (国内推荐)
    python download_objaverse_xl.py --mirror huggingface

    # 指定进程数
    python download_objaverse_xl.py --processes 8

参考文档:
- https://huggingface.co/datasets/allenai/objaverse-xl
- https://github.com/allenai/objaverse-xl
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# 检查 objaverse 是否安装
try:
    import objaverse.xl as oxl
except ImportError:
    print("错误: 请先安装 objaverse 包")
    print("运行: pip install objaverse")
    sys.exit(1)

# ============== 配置常量 ==============
DEFAULT_DOWNLOAD_DIR = "./data/objaverse"
DEFAULT_PROCESSES = None  # None 表示使用 CPU 核心数
PROGRESS_FILE = "download_progress.json"
FAILED_OBJECTS_FILE = "failed_objects.csv"
ANNOTATIONS_FILE = "annotations.parquet"
LOG_FILE = "download.log"

# 数据来源
SOURCES = ["github", "thingiverse", "sketchfab", "smithsonian"]

# 镜像站配置
# 各来源说明:
# - Sketchfab: 所有文件从 HuggingFace 下载，只需 HF 镜像
# - GitHub: 标注从 HF 下载，3D 文件通过 git clone 从 GitHub 下载
# - Thingiverse: 标注从 HF 下载，3D 文件从 thingiverse.com 下载
# - Smithsonian: 标注从 HF 下载，3D 文件从 3d-api.si.edu 下载
MIRRORS = {
    "none": {
        "name": "不使用镜像",
        "hf_endpoint": None,
        "git_mirror": None,
        "proxy": None,
        "description": "直接访问原始地址",
    },
    "huggingface": {
        "name": "HuggingFace 镜像",
        "hf_endpoint": "https://hf-mirror.com",
        "git_mirror": None,
        "proxy": None,
        "description": "仅 HuggingFace 镜像，适用于 Sketchfab 数据",
    },
    "china-full": {
        "name": "国内完整镜像 (推荐)",
        "hf_endpoint": "https://hf-mirror.com",
        "git_mirror": "https://ghproxy.com",
        "proxy": None,
        "description": "HuggingFace + GitHub 镜像，适用于 GitHub 数据",
    },
    "proxy": {
        "name": "使用代理",
        "hf_endpoint": None,
        "git_mirror": None,
        "proxy": "http://127.0.0.1:7890",  # 默认代理，可通过 --proxy 参数覆盖
        "description": "通过代理访问所有来源，适用于 Thingiverse/Smithsonian",
    },
}

# ============== 日志设置 ==============
def setup_logging(log_dir: Path) -> logging.Logger:
    """设置日志系统
    
    Args:
        log_dir: 日志文件目录
        
    Returns:
        配置好的 logger 实例
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_FILE
    
    # 创建 logger
    logger = logging.getLogger("objaverse_downloader")
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ============== 镜像站配置 ==============
def setup_mirror(
    mirror: str, 
    logger: logging.Logger, 
    proxy: Optional[str] = None
) -> None:
    """配置镜像站
    
    通过设置环境变量来实现镜像站访问
    
    Args:
        mirror: 镜像站名称
        logger: 日志记录器
        proxy: 可选的代理地址，覆盖配置中的默认代理
    """
    if mirror not in MIRRORS:
        logger.warning(f"未知镜像站: {mirror}，使用默认设置")
        return
    
    mirror_config = MIRRORS[mirror]
    
    if mirror == "none":
        logger.info("不使用镜像站，直接访问原始地址")
        return
    
    logger.info(f"启用镜像站: {mirror_config['name']}")
    logger.info(f"  说明: {mirror_config['description']}")
    
    # 设置 HuggingFace 镜像
    hf_endpoint = mirror_config.get("hf_endpoint")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        logger.info(f"  HF_ENDPOINT = {hf_endpoint}")
    
    # 设置 Git 镜像 (用于 GitHub 来源)
    git_mirror = mirror_config.get("git_mirror")
    if git_mirror:
        setup_git_mirror(git_mirror, logger)
    
    # 设置代理
    proxy_url = proxy or mirror_config.get("proxy")
    if proxy_url:
        setup_proxy(proxy_url, logger)


def setup_git_mirror(git_mirror: str, logger: logging.Logger) -> None:
    """配置 Git 镜像
    
    通过 git config 设置 URL 重写规则
    
    Args:
        git_mirror: Git 镜像地址
        logger: 日志记录器
    """
    import subprocess
    
    logger.info(f"  Git 镜像 = {git_mirror}")
    
    # 常见的 Git 镜像配置方式
    # 方式1: 使用 URL 重写 (ghproxy 等)
    if "ghproxy" in git_mirror or "mirror" in git_mirror:
        # 设置 git config url 重写
        try:
            # 格式: git config --global url."https://ghproxy.com/https://github.com/".insteadOf "https://github.com/"
            subprocess.run(
                ["git", "config", "--global", 
                 f"url.{git_mirror}/https://github.com/", 
                 "insteadOf", "https://github.com/"],
                check=True,
                capture_output=True
            )
            logger.info(f"    已配置 git url 重写: {git_mirror}/https://github.com/ -> https://github.com/")
        except subprocess.CalledProcessError as e:
            logger.warning(f"    配置 git 镜像失败: {e}")
    # 方式2: 使用其他镜像服务
    else:
        # 设置通用代理
        logger.info(f"    Git 镜像已配置: {git_mirror}")


def setup_proxy(proxy_url: str, logger: logging.Logger) -> None:
    """配置代理
    
    通过环境变量设置 HTTP/HTTPS 代理
    
    Args:
        proxy_url: 代理地址 (如 http://127.0.0.1:7890)
        logger: 日志记录器
    """
    logger.info(f"  代理 = {proxy_url}")
    
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url
    
    # 为 git 设置代理
    import subprocess
    try:
        subprocess.run(
            ["git", "config", "--global", "http.proxy", proxy_url],
            check=True,
            capture_output=True
        )
        subprocess.run(
            ["git", "config", "--global", "https.proxy", proxy_url],
            check=True,
            capture_output=True
        )
        logger.info(f"    已配置 git 代理: {proxy_url}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"    配置 git 代理失败: {e}")


# ============== 进度管理 ==============
class ProgressTracker:
    """下载进度跟踪器"""
    
    def __init__(self, progress_file: Path):
        self.progress_file = progress_file
        self.progress = self._load_progress()
    
    def _load_progress(self) -> Dict[str, Any]:
        """加载进度文件"""
        if self.progress_file.exists():
            with open(self.progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "start_time": None,
            "last_update": None,
            "total_objects": 0,
            "downloaded_objects": 0,
            "failed_objects": 0,
            "sources_completed": [],
        }
    
    def save_progress(self) -> None:
        """保存进度到文件"""
        self.progress["last_update"] = datetime.now().isoformat()
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)
    
    def start(self, total: int) -> None:
        """开始下载"""
        if self.progress["start_time"] is None:
            self.progress["start_time"] = datetime.now().isoformat()
        self.progress["total_objects"] = total
        self.save_progress()
    
    def update(self, downloaded: int, failed: int = 0) -> None:
        """更新进度"""
        self.progress["downloaded_objects"] = downloaded
        self.progress["failed_objects"] = failed
        self.save_progress()
    
    def mark_source_completed(self, source: str) -> None:
        """标记数据源完成"""
        if source not in self.progress["sources_completed"]:
            self.progress["sources_completed"].append(source)
        self.save_progress()


# ============== 失败记录 ==============
class FailedObjectsRecorder:
    """失败对象记录器"""
    
    def __init__(self, failed_file: Path):
        self.failed_file = failed_file
        self.failed_objects: List[Dict[str, str]] = []
        self._load_failed()
    
    def _load_failed(self) -> None:
        """加载已有的失败记录"""
        if self.failed_file.exists():
            with open(self.failed_file, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                self.failed_objects = list(reader)
    
    def add_failed(
        self, 
        file_identifier: str, 
        sha256: str, 
        source: str,
        error: str = ""
    ) -> None:
        """添加失败记录"""
        self.failed_objects.append({
            "file_identifier": file_identifier,
            "sha256": sha256,
            "source": source,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_failed()
    
    def _save_failed(self) -> None:
        """保存失败记录"""
        with open(self.failed_file, "w", encoding="utf-8", newline="") as f:
            fieldnames = ["file_identifier", "sha256", "source", "error", "timestamp"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.failed_objects)
    
    @property
    def count(self) -> int:
        return len(self.failed_objects)


# ============== 标注管理 ==============
def download_annotations(download_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    """下载标注数据
    
    Args:
        download_dir: 下载目录
        logger: 日志记录器
        
    Returns:
        标注 DataFrame
    """
    logger.info("开始下载标注数据...")
    
    # 使用 objaverse API 下载标注
    annotations = oxl.get_annotations(download_dir=str(download_dir))
    
    # 保存到 parquet 文件
    annotations_file = download_dir / ANNOTATIONS_FILE
    annotations.to_parquet(annotations_file, index=False)
    
    logger.info(f"标注数据已保存到: {annotations_file}")
    logger.info(f"总对象数: {len(annotations):,}")
    
    # 打印统计信息
    logger.info("数据来源分布:")
    for source, count in annotations["source"].value_counts().items():
        logger.info(f"  - {source}: {count:,}")
    
    logger.info("文件类型分布:")
    for file_type, count in annotations["fileType"].value_counts().items():
        logger.info(f"  - {file_type}: {count:,}")
    
    return annotations


def load_annotations(download_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    """加载标注数据
    
    优先从本地 parquet 文件加载，如果不存在则从缓存目录加载
    
    Args:
        download_dir: 下载目录
        logger: 日志记录器
        
    Returns:
        标注 DataFrame
    """
    annotations_file = download_dir / ANNOTATIONS_FILE
    
    if annotations_file.exists():
        logger.info(f"从本地加载标注数据: {annotations_file}")
        return pd.read_parquet(annotations_file)
    
    # 尝试从 objaverse 缓存加载
    logger.info("从 objaverse 缓存加载标注数据...")
    return oxl.get_annotations(download_dir=str(download_dir))


# ============== 下载回调 ==============
def create_callbacks(
    logger: logging.Logger,
    progress_tracker: ProgressTracker,
    failed_recorder: FailedObjectsRecorder,
    downloaded_count: List[int],  # 使用 list 作为可变引用
):
    """创建下载回调函数
    
    Args:
        logger: 日志记录器
        progress_tracker: 进度跟踪器
        failed_recorder: 失败记录器
        downloaded_count: 已下载计数器
        
    Returns:
        回调函数字典
    """
    
    def handle_found_object(
        local_path: str,
        file_identifier: str,
        sha256: str,
        metadata: Dict[str, Any]
    ) -> None:
        """处理成功下载的对象"""
        downloaded_count[0] += 1
        if downloaded_count[0] % 100 == 0:
            logger.info(f"已下载: {downloaded_count[0]:,} 个对象")
            progress_tracker.update(downloaded_count[0], failed_recorder.count)
    
    def handle_modified_object(
        local_path: str,
        file_identifier: str,
        new_sha256: str,
        old_sha256: str,
        metadata: Dict[str, Any]
    ) -> None:
        """处理内容变更的对象"""
        logger.warning(f"对象内容变更: {file_identifier}")
        logger.warning(f"  旧 SHA256: {old_sha256[:16]}...")
        logger.warning(f"  新 SHA256: {new_sha256[:16]}...")
        downloaded_count[0] += 1
    
    def handle_missing_object(
        file_identifier: str,
        sha256: str,
        metadata: Dict[str, Any]
    ) -> None:
        """处理缺失的对象"""
        source = metadata.get("source", "unknown")
        failed_recorder.add_failed(file_identifier, sha256, source, "对象不存在")
        logger.warning(f"对象不存在: {file_identifier}")
    
    def handle_new_object(
        local_path: str,
        file_identifier: str,
        sha256: str,
        metadata: Dict[str, Any]
    ) -> None:
        """处理新发现的对象 (仅 GitHub)"""
        pass  # 不做特殊处理
    
    return {
        "handle_found_object": handle_found_object,
        "handle_modified_object": handle_modified_object,
        "handle_missing_object": handle_missing_object,
        "handle_new_object": handle_new_object,
    }


# ============== 批量下载 ==============
def download_objects(
    annotations: pd.DataFrame,
    download_dir: Path,
    processes: Optional[int],
    logger: logging.Logger,
    progress_tracker: ProgressTracker,
    failed_recorder: FailedObjectsRecorder,
) -> Dict[str, str]:
    """批量下载 3D 对象
    
    Args:
        annotations: 标注 DataFrame
        download_dir: 下载目录
        processes: 并行进程数
        logger: 日志记录器
        progress_tracker: 进度跟踪器
        failed_recorder: 失败记录器
        
    Returns:
        下载结果字典 {file_identifier: local_path}
    """
    logger.info(f"开始下载 {len(annotations):,} 个对象...")
    logger.info(f"并行进程数: {processes or os.cpu_count()}")
    
    # 初始化进度
    progress_tracker.start(len(annotations))
    
    # 计数器 (使用 list 作为可变引用)
    downloaded_count = [0]
    
    # 创建回调函数
    callbacks = create_callbacks(
        logger, 
        progress_tracker, 
        failed_recorder,
        downloaded_count
    )
    
    # 执行下载
    result = oxl.download_objects(
        objects=annotations,
        download_dir=str(download_dir),
        processes=processes,
        handle_found_object=callbacks["handle_found_object"],
        handle_modified_object=callbacks["handle_modified_object"],
        handle_missing_object=callbacks["handle_missing_object"],
        handle_new_object=callbacks["handle_new_object"],
    )
    
    # 更新最终进度
    progress_tracker.update(len(result), failed_recorder.count)
    
    logger.info(f"下载完成! 成功: {len(result):,}, 失败: {failed_recorder.count:,}")
    
    return result


def download_by_source(
    annotations: pd.DataFrame,
    download_dir: Path,
    processes: Optional[int],
    logger: logging.Logger,
    progress_tracker: ProgressTracker,
    failed_recorder: FailedObjectsRecorder,
    sources: Optional[List[str]] = None,
) -> Dict[str, str]:
    """按数据源分批下载
    
    这样可以更好地控制下载过程，并支持按源恢复
    
    Args:
        annotations: 标注 DataFrame
        download_dir: 下载目录
        processes: 并行进程数
        logger: 日志记录器
        progress_tracker: 进度跟踪器
        failed_recorder: 失败记录器
        sources: 要下载的数据源列表，None 表示全部
        
    Returns:
        下载结果字典
    """
    all_results = {}
    sources_to_download = sources or SOURCES
    
    for source in sources_to_download:
        # 跳过已完成的源
        if source in progress_tracker.progress["sources_completed"]:
            logger.info(f"跳过已完成的源: {source}")
            continue
        
        logger.info(f"\n{'='*50}")
        logger.info(f"开始下载 {source} 数据...")
        logger.info(f"{'='*50}")
        
        # 过滤当前源的对象
        source_annotations = annotations[annotations["source"] == source]
        
        if len(source_annotations) == 0:
            logger.warning(f"没有找到 {source} 的对象")
            continue
        
        logger.info(f"{source} 对象数量: {len(source_annotations):,}")
        
        # 下载
        result = download_objects(
            source_annotations,
            download_dir,
            processes,
            logger,
            progress_tracker,
            failed_recorder,
        )
        
        all_results.update(result)
        
        # 标记完成
        progress_tracker.mark_source_completed(source)
        
        logger.info(f"{source} 下载完成，本源成功: {len(result):,}")
    
    return all_results


# ============== 主函数 ==============
def main():
    parser = argparse.ArgumentParser(
        description="Objaverse-XL 数据集下载脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 完整下载
    python download_objaverse_xl.py

    # 测试下载 (随机采样 100 个对象)
    python download_objaverse_xl.py --sample_size 100

    # 只下载特定来源
    python download_objaverse_xl.py --sources sketchfab smithsonian

    # 指定并行进程数
    python download_objaverse_xl.py --processes 8

    # 只下载标注，不下载文件
    python download_objaverse_xl.py --annotations_only

    # 使用已缓存的标注
    python download_objaverse_xl.py --skip_annotations

镜像站配置:
    # 不使用镜像 (默认)
    python download_objaverse_xl.py --mirror none

    # HuggingFace 镜像 (适用于 Sketchfab 数据)
    python download_objaverse_xl.py --mirror huggingface

    # 国内完整镜像 (HuggingFace + GitHub，推荐)
    python download_objaverse_xl.py --mirror china-full

    # 使用代理 (适用于 Thingiverse/Smithsonian)
    python download_objaverse_xl.py --mirror proxy --proxy http://127.0.0.1:7890

    # 自定义代理地址
    python download_objaverse_xl.py --mirror none --proxy socks5://127.0.0.1:1080

    # 测试 + 国内完整镜像
    python download_objaverse_xl.py --sample_size 100 --mirror china-full

数据来源说明:
    - sketchfab: 仅需 HuggingFace 镜像
    - github: 需要 HuggingFace + GitHub 镜像 (使用 china-full)
    - thingiverse: 需要代理访问 thingiverse.com (使用 proxy)
    - smithsonian: 需要代理访问 3d-api.si.edu (使用 proxy)
        """
    )
    
    parser.add_argument(
        "--download_dir",
        type=str,
        default=DEFAULT_DOWNLOAD_DIR,
        help=f"下载目录 (默认: {DEFAULT_DOWNLOAD_DIR})"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="并行下载进程数 (默认: CPU 核心数)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="采样数量 (用于测试，None 表示全部)"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=SOURCES,
        default=None,
        help=f"要下载的数据源 (默认: 全部)"
    )
    parser.add_argument(
        "--annotations_only",
        action="store_true",
        help="只下载标注，不下载文件"
    )
    parser.add_argument(
        "--skip_annotations",
        action="store_true",
        help="跳过标注下载，使用缓存"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (用于采样)"
    )
    parser.add_argument(
        "--mirror",
        type=str,
        choices=list(MIRRORS.keys()),
        default="none",
        help=f"镜像站选择 (默认: none)。可选: {list(MIRRORS.keys())}"
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="代理地址，覆盖镜像配置中的默认代理 (如: http://127.0.0.1:7890)"
    )
    
    args = parser.parse_args()
    
    # 创建下载目录
    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(download_dir)
    
    # 配置镜像站
    setup_mirror(args.mirror, logger, args.proxy)
    
    logger.info("="*60)
    logger.info("Objaverse-XL 数据集下载脚本")
    logger.info("="*60)
    logger.info(f"下载目录: {download_dir.absolute()}")
    logger.info(f"并行进程: {args.processes or os.cpu_count()}")
    if args.sources:
        logger.info(f"数据源: {args.sources}")
    if args.sample_size:
        logger.info(f"采样数量: {args.sample_size}")
    
    # 初始化进度跟踪和失败记录
    progress_tracker = ProgressTracker(download_dir / PROGRESS_FILE)
    failed_recorder = FailedObjectsRecorder(download_dir / FAILED_OBJECTS_FILE)
    
    # 下载/加载标注
    if args.skip_annotations:
        annotations = load_annotations(download_dir, logger)
    else:
        annotations = download_annotations(download_dir, logger)
    
    # 如果只需要标注，退出
    if args.annotations_only:
        logger.info("标注下载完成，退出")
        return
    
    # 采样
    if args.sample_size:
        logger.info(f"随机采样 {args.sample_size} 个对象...")
        annotations = annotations.sample(n=args.sample_size, random_state=args.seed)
        logger.info(f"采样后对象数: {len(annotations):,}")
    
    # 过滤数据源
    if args.sources:
        annotations = annotations[annotations["source"].isin(args.sources)]
        logger.info(f"过滤后对象数: {len(annotations):,}")
    
    # 执行下载
    start_time = datetime.now()
    
    try:
        if args.sources and len(args.sources) == 1:
            # 单一来源，直接下载
            result = download_objects(
                annotations,
                download_dir,
                args.processes,
                logger,
                progress_tracker,
                failed_recorder,
            )
        else:
            # 多来源，按源分批下载
            result = download_by_source(
                annotations,
                download_dir,
                args.processes,
                logger,
                progress_tracker,
                failed_recorder,
                args.sources,
            )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info("下载统计")
        logger.info("="*60)
        logger.info(f"总下载对象: {len(result):,}")
        logger.info(f"失败对象: {failed_recorder.count:,}")
        logger.info(f"耗时: {duration}")
        logger.info(f"平均速度: {len(result) / duration.total_seconds():.2f} 对象/秒")
        
        # 保存下载结果
        results_file = download_dir / "download_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({
                "total_downloaded": len(result),
                "failed_count": failed_recorder.count,
                "duration_seconds": duration.total_seconds(),
                "download_time": end_time.isoformat(),
            }, f, indent=2)
        
        logger.info(f"下载结果已保存到: {results_file}")
        
    except KeyboardInterrupt:
        logger.warning("\n用户中断下载")
        logger.info(f"进度已保存，可重新运行继续下载")
    except Exception as e:
        logger.error(f"下载出错: {e}")
        raise


if __name__ == "__main__":
    main()
