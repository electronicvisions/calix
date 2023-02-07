from pathlib import Path
import pickle
import tempfile
import errno
from hashlib import sha256
import os
from typing import List, Optional
from dlens_vx_v3 import hxcomm, sta, logger
import calix.hagen
import calix.spiking
from calix.common import base


def _calibrate(
    connection,
    target: base.CalibTarget,
    options: Optional[base.CalibOptions] = None,
) -> base.CalibResult:
    # TODO: find more elegant way than mass if else statements
    if isinstance(target, calix.spiking.SpikingCalibTarget):
        if options is None:
            options = calix.spiking.SpikingCalibOptions()
        elif not isinstance(options, calix.spiking.SpikingCalibOptions):
            raise TypeError("Provided target and option types do not match")
        calib_func = calix.spiking.calibrate
    elif isinstance(target, calix.hagen.HagenCalibTarget):
        if options is None:
            options = calix.hagen.HagenCalibOptions()
        elif not isinstance(options, calix.hagen.HagenCalibOptions):
            raise TypeError("Provided target and option types do not match")
        calib_func = calix.hagen.calibrate
    elif isinstance(target, calix.hagen.HagenSyninCalibTarget):
        if options is None:
            options = calix.hagen.HagenSyninCalibOptions()
        elif not isinstance(options, calix.hagen.HagenSyninCalibOptions):
            raise TypeError("Provided target and option types do not match")
        calib_func = calix.hagen.calibrate_for_synin_integration
    else:
        raise NotImplementedError(f"Target of type {target} not supported")

    builder, _ = sta.generate(sta.ExperimentInit())
    sta.run(connection, builder.done())
    return calib_func(connection, target, options)


def calibrate(
    target: base.CalibTarget,
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
        if os.getenv("XDG_CACHE_HOME") is not None:
            localcache = Path(os.getenv("XDG_CACHE_HOME")).joinpath("calix")
        else:
            localcache = Path.home().joinpath(".cache/calix")

        cache_paths = [Path("/wang/data/calibration/hicann-dls-sr-hx/cache"),
                       localcache]

    log = logger.get("calix.calibrate")

    if connection is None:
        conn_manager = hxcomm.ManagedConnection()
        connection = conn_manager.__enter__()

    # The key into the cache is defined by the parameters target, options.
    # As the data-holding parameters are mutable, we define a custom hashing
    # function
    # TODO: incorporate calix software state in hash.
    str_to_hash = repr(target) + repr(options) +\
        repr(connection.get_unique_identifier())
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
