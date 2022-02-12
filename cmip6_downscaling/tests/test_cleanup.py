from adlfs import AzureBlobFileSystem

from cmip6_downscaling.tasks.cleanup import instantiate_az_filesystem  # , remove_stores


def test_instantiate_az_filesystem():
    filesystem = instantiate_az_filesystem()
    assert isinstance(filesystem, AzureBlobFileSystem)


# perhaps a comprehensive test of this would be to do a tiny run with the cleanup flag on and then confirm that an fs.ls of the bucket at the end returns a FileNotFoundError?
# def test_remove_stores():
#     """create tiny recursive directory, try removing with util, assert dir does not exist"""
#     filesystem = instantiate_az_filesystem()
#     tmp_fs_name = config.get("storage.temporary.uri") + "/test_temp_dir"
#     filesystem.mkdir(tmp_fs_name)
#     remove_stores([tmp_fs_name])
#     assert filesystem(tmp_fs_name).exists()
