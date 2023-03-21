from pathlib import Path
import pickle
import tempfile
import errno
from hashlib import sha256
import os
from typing import List, Optional
from dlens_vx_v3 import hxcomm, sta, logger
import pyccalix
from calix.common import base

_DEFAULT_GLOBAL_CACHE: Path = Path("/wang",
                                   "data",
                                   "calibration",
                                   "hicann-dls-sr-hx",
                                   "cache")

_DEFAULT_LOCAL_CACHE: Path
if os.getenv("XDG_CACHE_HOME") is not None:
    _DEFAULT_LOCAL_CACHE = Path(os.getenv("XDG_CACHE_HOME")).joinpath("calix")
else:
    _DEFAULT_LOCAL_CACHE = Path.home().joinpath(".cache", "calix")


def _calibrate(
    connection,
    target: base.TopLevelCalibTarget,
    options: Optional[base.CalibOptions] = None,
) -> base.CalibResult:
    builder, _ = sta.generate(sta.ExperimentInit())
    sta.run(connection, builder.done())
    return target.calibrate(connection, options)


def calibrate(
    target: base.TopLevelCalibTarget,
    options: Optional[base.CalibOptions] = None,
    cache_paths: Optional[List[Path]] = None,
    cache_read_only: Optional[bool] = False,
    connection: Optional = None
) -> base.CalibResult:
    """
    Calibrate chip with cache functionality. Calibration function is deduced
    from target and options type. If cache_paths is not empty, searches for
    cached calibration result with same parameters in list order. If no cache
    is found executes calibration and caches in fist path with write access.

    :param target: Target values for calibration.
    :param options: Options for calibration.
    :param cache_paths: List of possible cache locations. If list is empty,
                        caching is skipped. If None defaults are used. Defaults
                        are read-only shared wang cache path followed by user
                        home cache.
    :param cache_read_only: Only read cache file, do not create cache file if
                            one does not exist.
    :return: Calibration result
    """
    if cache_paths is None:
        cache_paths = [_DEFAULT_GLOBAL_CACHE, _DEFAULT_LOCAL_CACHE]

    log = logger.get("calix.calibrate")

    if connection is None:
        conn_manager = hxcomm.ManagedConnection()
        connection = conn_manager.__enter__()

    if len(cache_paths) == 0:
        log.info("List of calib cache paths is empty, cache disabled.")
        return _calibrate(connection, target, options)

    # The key into the cache is defined by the parameters target, options.
    # As the data-holding parameters are mutable, we define a custom hashing
    # function
    str_to_hash = repr(target) + repr(options) +\
        repr(connection.get_unique_identifier()) +\
        repr(pyccalix.helpers.get_repo_state())
    filename = str(sha256(str_to_hash.encode("utf-8")).hexdigest())

    # search for existing cache
    for path in cache_paths:

        cache_file_path = path.joinpath(filename)
        if not cache_file_path.is_file():
            continue

        log.INFO(f"Applying cached calibration in {cache_file_path}")
        with cache_file_path.open(mode="rb") as myfile:
            return pickle.load(myfile)

    # no cache found
    log.INFO("Executing calibration")
    result = _calibrate(connection, target, options)

    if cache_read_only:
        return result

    # create cache
    for path in cache_paths:
        path.mkdir(parents=True, exist_ok=True)
        cache_file_path = path.joinpath(filename)
        # pickle result file in temp file and rename due to nfs
        try:
            with tempfile.NamedTemporaryFile(dir=path, delete=False) as myfile:
                pickle.dump(result, myfile)
                myfile.close()
                if cache_file_path.exists():
                    log.WARN(f"Cache file {cache_file_path} was created by"
                             "other process in meantime")
                    return result
                try:
                    os.rename(myfile.name, cache_file_path)
                except OSError as err:
                    # try next path if no write permission
                    if err.errno == errno.EACCES:
                        continue
                    raise err
            log.INFO(f"Cached result in {cache_file_path}")
        except PermissionError:
            # In case of no write permission try next path
            continue
        return result

    raise RuntimeError("Could not create cache file in any "
                       f"directory: {cache_paths}")
