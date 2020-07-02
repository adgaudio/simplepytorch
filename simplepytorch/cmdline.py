"""
Tooling to initialize and run models from commandline
"""
import abc
import configargparse as ap
import importlib
from os.path import basename, dirname, isdir, join
import logging
import shlex
import sys


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def load_model_config(kls=None, optional_cmdline_string: str=None):
    """Programmatically initialize a model config class with its defaults

    :kls: is your subclass implementation of a FeedForwardModelConfig.
    :optional_cmdline_string:  Arguments you would pass at commandline, if any.
    :returns: kls() initialized with default parameters.

    Example:
        # programmatically initialize your model config class
        from examples import LetsTrainSomething  # your class here
        load_model_config(LetsTrainSomething, '--epochs 3')

        # or 
        load_model_config(LetsTrainSomething)
    """
    if kls is None:
        if optional_cmdline_string:
            ns = build_arg_parser().parse_args(
                shlex.split(optional_cmdline_string))
        else:
            ns = build_arg_parser().parse_args()
    else:
        fp = sys.modules[kls.__module__].__file__
        ns = build_arg_parser(fp).parse_args(
            [kls.__name__] + shlex.split(optional_cmdline_string or ''))
    # merge cmdline config with defaults
    config_overrides = ns.__dict__
    config = config_overrides.pop('modelconfig_class')(config_overrides)

    log.info('\n'.join(
        str((k, v)) for k, v in config.__dict__.items()
        if not k.startswith('_')))

    return config


def main():
    """Initialize model and run from command-line"""
    # --> logging initialization
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.getLogger(__name__.split('.')[0]).setLevel(logging.INFO)
    # --> initialize model configuration
    config = load_model_config()
    # --> start training or run your code
    config.run()


def _add_subparser_find_configurable_attributes(kls):
    """
    Examine a class and all ancestor classes to return dict of form:
    {grandparent_class: [class variables, ...],
     parent_class: [unseen class_variables, ...],
     class: [unseen class_variables, ...]
    }
    # get a list of configurable attributes from given class and parent classes
    """
    keys = dict()  # klass: set(keys...)
    klass_seen = [kls]
    klass_lst = [kls]
    while klass_lst:
        klass = klass_lst.pop()
        for k in klass.__bases__:
            if k in klass_seen:
                continue
            klass_seen.append(k)
            klass_lst.append(k)
    options_seen = set()
    for klass in reversed(klass_seen):
        # populate list of class variables
        tmp = {x for x in klass.__dict__ if not x.startswith('_')}
        # --> include CmdlineOptions that were defined like:
        #     __something = CmdlineOptions(...)
        tmp.update({
            x for x in klass.__dict__ if x.startswith('_%s__' % klass.__name__)
            and isinstance(getattr(klass, x), CmdlineOptions)})
        # add keys that weren't previously seen
        keys[klass] = tmp.difference(options_seen)
        # add CmdlineOptions that redefine some class variable.
        keys[klass].update({
            k for k in options_seen.intersection(tmp)
            if isinstance(getattr(klass, k), CmdlineOptions)})
        options_seen.update(keys[klass])
    return keys


def _add_subparser_arg(subparser, k, v, choices=None):
    """Add an argparse option via subparser.add_argument(...)
    for the given attribute k, with default value v
    `in_dict` (str) - if given, create a dictionary at returned
    argparse.Namespace.STR where STR is value of `in_dict`.  and add the parsed
    option to this dictionary
    """
    g = subparser
    accepted_simple_types = (int, float, str)
    ku = k.replace('_', '-')
    if isinstance(v, bool):
        grp = g.add_mutually_exclusive_group()
        grp.add_argument(
            '--%s' % ku, action='store_const', const=True, default=v,
            env_var=k)
        grp.add_argument(
            '--no-%s' % ku, action='store_const', const=False, dest=k,
            env_var=k)
    elif isinstance(v, accepted_simple_types):
        g.add_argument(
            '--%s' % ku, type=type(v), default=v, help=' ', env_var=k, choices=choices)
    elif isinstance(v, (list, tuple)):
        if all(isinstance(x, accepted_simple_types) for x in v):
            g.add_argument(
                '--%s' % ku, nargs=len(v), default=v, help=' ', env_var=k,
                type=lambda inpt: type(accepted_simple_types)(
                    [typ(x) for typ, x in zip(accepted_simple_types, inpt)])[0],
                choices=choices)
        else:
            g.add_argument('--%s' % ku, nargs=len(v), type=v[0], env_var=k, choices=choices)
    elif any(v is x for x in accepted_simple_types):
        g.add_argument('--%s' % ku, type=v, env_var=k, required=True, choices=choices)


def add_subparser(subparsers, name: str, modelconfig_class: type):
    """
    Automatically add parser options for attributes in given class and
    all of its parent classes
    """
    g = subparsers.add_parser(
        #  name, formatter_class=ap.RawDescriptionHelpFormatter)
        name, formatter_class=ap.ArgumentDefaultsHelpFormatter)
    g.add_argument(
        '--modelconfig_class', help=ap.SUPPRESS, default=modelconfig_class)

    # add an argument for each configurable key that we can work with
    keys = _add_subparser_find_configurable_attributes(modelconfig_class)
    grps = {}
    # in help message, populate the order in which subparsers appear
    for klass in keys:
        if not keys[klass]: continue
        grp = g.add_argument_group("Options from class %s" % klass.__name__,
                                   description=klass.__doc__)
        grps[klass] = grp
    # in help message, add options under each subparser.
    cmdline_options_already_added = set()
    for klass in reversed(list(grps.keys())):
        grp = grps[klass]
        for k in keys[klass]:
            v = getattr(modelconfig_class, k)
            if isinstance(v, CmdlineOptions):
                if v._id in grps:
                    grp2 = grps[v._id]
                else:
                    grp2 = g.add_argument_group(
                        #  "Options for %s -> %s" % (klass.__name__, v._id),
                        "Options for %s" % v._id,
                        description="Manually defined options")
                    grps[v._id] = grp2
                for k2, v2 in v._params.items():
                    tmpkey = '%s_%s' % (v._id.lower(), k2)
                    if tmpkey in cmdline_options_already_added:
                        continue
                    cmdline_options_already_added.add(tmpkey)
                    _add_subparser_arg(grp2, tmpkey, v2, choices=v._choices.get(k2))

            else:
                _add_subparser_arg(grp, k, v)


class ModelConfigABC(abc.ABC):
    """All model configs should inherit from this class to get recognized on the
    commandline."""
    pass


class CmdlineOptions:
    """
    Model config classes (or parent classes) may wish to expose a group of
    parameters.

    Instantiate this class in a model config (or parent class)

        >>> class MyModelConfig:
            ignored_variable_name = CmdlineOptions(
              'optimizer', {'weight_decay': float, 'lr': float}
              choices={'lr': [0.1, 0.2]})  # choices restricts possible lr values

    Then the params get expose on cmdline like this:

        $ python ... --optimizer-weight-decay 0.001 --optimizer-lr .0001
    """
    def __init__(self, id, params, choices=None):
        self._id = id
        self._params = params
        self._choices = choices or {}

    def kwargs(self, config):
        """Convenience function to collect the params stored in config object
        and return a dict"""
        return {k: getattr(config, '%s_%s' % (self._id, k))
                for k in self._params}

    def __repr__(self):
        return 'CmdlineOptions:%s' % self._id


def dynamically_import_model_configs(fp):
    if fp.endswith('/__init__.py'):
        fp = fp[:-12]
    fp = fp.rstrip('/')
    if isdir(fp):
        _name = basename(fp)
        fp = join(fp, '__init__.py')
    else:
        _name = basename(fp)[:-3]
    _fp = fp
    print(fp, _name)
    spec = importlib.util.spec_from_file_location(
        _name, fp, submodule_search_locations=['.'] +
        [_fp:=dirname(_fp) for _ in range(_fp.count('/'))])
        #  basename(fp)[:-3], fp, submodule_search_locations=['.'] + [_fp:=dirname(_fp) for _ in range(_fp.count('/'))])
    if spec is None:
        print(f"Unrecognized filepath: {fp}")
        sys.exit(1)
    MC = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = MC
    spec.loader.exec_module(MC)
    return MC


def build_arg_parser(fp=None):
    """Returns a parser to handle command-line arguments

    `fp` - The filepath to the python file containing Model Configs.
    The parser finds cmdline options by analyzing the file.
    By default, fp is assumed to be passed in as the first argument on cmdline.
    It is useful to pass `fp` yourself if you are programmatically
    initializing a class and with to ignore command-line arguments.
    """
    p = ap.ArgumentParser(
        formatter_class=ap.ArgumentDefaultsHelpFormatter)
    sp = p.add_subparsers(help='The model configuration to work with')
    sp.required = True
    sp.dest = 'model_configuration_name'

    if fp is None:
        try:
            fp = sys.argv.pop(1)
            #  kls = sys.argv[2]
            assert fp not in ['-h', '--help']
        except IndexError:
            print("usage:  <filepath> \nplease pass filepath to Python module containing your model config class")
            sys.exit(1)
    MC = dynamically_import_model_configs(fp)

    for kls_name in dir(MC):
        if kls_name.startswith("_"):
            continue
        # add all available model config classes as command line options
        mc_obj = getattr(MC, kls_name)
        if isinstance(mc_obj, type) and issubclass(mc_obj, ModelConfigABC):
            add_subparser(sp, mc_obj.__name__, mc_obj)
    return p
