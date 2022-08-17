#!/usr/bin/env python
# coding=utf-8
import os

from setuptools import find_packages
from setuptools import setup
from setuptools import Extension
import distutils.sysconfig
distutils.sysconfig.get_config_var("LINKFORSHARED")


def version_func(b):
    return "1.0."


def local_version_func(b):
    build_num = os.getenv("BUILD_NUMBER")
    if not build_num:
        return '999999'
    else:
        return build_num


def get_requirements_txt():
    result = os.popen("nvidia-smi | grep \"CUDA Version:\"")
    lines = [line for line in result.readlines()]
    for line in lines:
        elements = line.split("CUDA Version:")
        r = elements[1].strip().split(' ')
        main_version = int(r[0].split('.')[0])
        if main_version == 11:
            return "requirements-cu11.txt"
        return "requirements.txt"


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists
    import re
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


PYBIND_PATH = os.path.join(os.environ['VENV_PATH'], "lib/python3.8/site-packages/pybind11/include")

segmentation = Extension(name='segmentation',
                               sources=[
                                        'lib/grndseg/bin.cc',
                                        'lib/grndseg/segment.cc',
                                        'lib/grndseg/ground_segmentation.cc',
                                        'lib/grndseg/segmentation.cc'
                                        ],
                               include_dirs=[r'/usr/include/eigen3/', PYBIND_PATH, r'/usr/include/pcl-1.8/',
                                            r'/usr/include/', r'/usr/include/vtk-6.3/'],

                                libraries=['pcl_apps',
                                           'pcl_features',
                                           'pcl_filters',
                                           'pcl_io_ply',
                                           'pcl_io',
                                           'pcl_keypoints',
                                           'pcl_ml',
                                           'pcl_octree',
                                           'pcl_segmentation',
                                           'pcl_visualization',
                                           'pcl_sample_consensus'],
                                library_dirs=['/usr/lib/x86_64-linux-gnu/'],
                                extra_link_args=['-export-dynamic'])


requirements_fname = "requirements.txt"
setup(
    name="ff",
    author="chenan",
    author_email="ff@github.com",
    url=os.environ.get("GIT_FULL_URL", "https://github.com/peiyunh/ff"),
    description="Safe Local Motion Planning with Self-Supervised Freespace Forecasting",
    long_description=open("README.md", 'r').read().strip(),
    packages=find_packages(exclude=['analysis*']),
    setup_requires=['setuptools_scm'],
    use_scm_version={
        'version_scheme': version_func,
        'local_scheme': local_version_func,
        'write_to': "version.py",
    },
    install_requires=parse_requirements(requirements_fname),
    extras_require={
        'all': parse_requirements(requirements_fname),
        'tests': parse_requirements('test-requirements.txt'),
    },
    python_requires='>=3',
    ext_modules=[segmentation]
)

