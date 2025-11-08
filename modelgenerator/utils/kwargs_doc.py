import os
import inspect

from inspect import getfullargspec, unwrap
from typing import Any, Callable
from typing import ClassVar
from types import FunctionType
from types import WrapperDescriptorType

from docstring_inheritance.docstring_inheritors.bases import SectionsType
from docstring_inheritance.docstring_inheritors.bases.inheritor import (
    get_similarity_ratio,
    BaseDocstringInheritor,
)
from docstring_inheritance.docstring_inheritors.google import (
    DocstringParser,
    DocstringRenderer,
)

DocstringInheritorClass = type[BaseDocstringInheritor]


class ClassDocstringsInheritor:
    """A class for inheriting class docstrings. This class is borrowed from
    `docstring_inheritance/class_docstrings_inheritor.py` and modified to work with kwargs docstrings and update the
    signature of the child class."""

    _cls: type
    """The class to process."""

    _docstring_inheritor: DocstringInheritorClass
    """The docstring inheritor."""

    _init_in_class: bool
    """Whether the ``__init__`` arguments documentation is in the class docstring."""

    __mro_classes: list[type]
    """The MRO classes."""

    def __init__(
        self,
        cls: type,
        docstring_inheritor: DocstringInheritorClass,
        init_in_class: bool,
    ) -> None:
        """
        Args:
            cls: The class to process.
            docstring_inheritor: The docstring inheritor.
            init_in_class: Whether the ``__init__`` arguments documentation is in the
                class docstring.
        """  # noqa: D205, D212
        # Remove the new class itself and the object class from the mro,
        # object's docstrings have no interest.
        self.__mro_classes = cls.mro()[1:-1]
        self._cls = cls
        self._docstring_inheritor = docstring_inheritor
        self._init_in_class = init_in_class

    @classmethod
    def inherit_docstrings(
        cls,
        class_: type,
        docstring_inheritor: DocstringInheritorClass,
        init_in_class: bool,
    ) -> None:
        """Inherit all the docstrings of the class.

        Args:
            class_: The class to process.
            docstring_inheritor: The docstring inheritor.
            init_in_class: Whether the ``__init__`` arguments documentation is in the
                class docstring.
        """
        inheritor = cls(class_, docstring_inheritor, init_in_class)
        inheritor._inherit_attrs_docstrings()
        inheritor._inherit_class_docstring()

    def _inherit_class_docstring(
        self,
    ) -> None:
        """Create the inherited docstring for the class docstring."""
        func = None
        old_init_doc = None
        init_doc_changed = False

        if self._init_in_class:
            init_method: Callable[..., None] = self._cls.__init__  # type: ignore[misc]
            # Ignore the case when __init__ is from object since there is no docstring
            # and its __doc__ cannot be assigned.
            if not isinstance(init_method, WrapperDescriptorType):
                old_init_doc = init_method.__doc__
                init_method.__doc__ = self._cls.__doc__
                func = init_method
                init_doc_changed = True

        if func is None:
            func = self._create_dummy_func_with_doc(self._cls.__doc__)

        for parent_cls in self.__mro_classes:
            # As opposed to the attribute inheritance, and following the way a class is
            # assembled by type(), the docstring of a class is the combination of the
            # docstrings of its parents.
            self._docstring_inheritor.inherit(
                parent_cls.__doc__, func, child_func_signature_updated=True
            )

        self._cls.__doc__ = func.__doc__

        if self._init_in_class and init_doc_changed:
            init_method.__doc__ = old_init_doc

    def _inherit_attrs_docstrings(
        self,
    ) -> None:
        """Create the inherited docstrings for the class attributes."""
        for attr_name, attr in self._cls.__dict__.items():
            if not isinstance(attr, FunctionType):
                continue

            for parent_cls in self.__mro_classes:
                parent_method = getattr(parent_cls, attr_name, None)
                if parent_method is not None:
                    parent_doc = parent_method.__doc__
                    if parent_doc is not None:
                        self._docstring_inheritor.inherit(parent_doc, attr, parent_method)
                        # As opposed to the class docstring inheritance, and following
                        # the MRO for methods,
                        # we inherit only from the first found parent.
                        break
                    # TODO: else WARN that no docstring is defined and
                    # none can be inherited.

    @staticmethod
    def _create_dummy_func_with_doc(docstring: str | None) -> Callable[..., Any]:
        """Create a dummy function with a given docstring.

        Args:
            docstring: The docstring to be assigned.

        Returns:
            The function with the given docstring.
        """

        def func() -> None:  # pragma: no cover
            pass

        func.__doc__ = docstring
        return func


class _BaseDocstringInheritanceMeta(type):
    """Base metaclass for inheriting class docstrings. This class is borrowed from docstring_inheritance/__init__.py"""

    def __init__(
        cls,
        class_name: str,
        class_bases: tuple[type],
        class_dict: dict[str, Any],
        docstring_inheritor: DocstringInheritorClass,
        init_in_class: bool,
    ) -> None:
        super().__init__(class_name, class_bases, class_dict)
        if class_bases:
            ClassDocstringsInheritor.inherit_docstrings(cls, docstring_inheritor, init_in_class)


class GoogleKwargsDocstringInheritor(BaseDocstringInheritor):
    """The inheritor for Google docstrings with kwargs. This class is used to inherit all args in the parent docstring
    if the child docstring has kwargs."""

    _MISSING_ARG_TEXT = f": {BaseDocstringInheritor.MISSING_ARG_DESCRIPTION}"
    _INHERIT_ARG_SUFFIX = " (Keyword argument)"
    _DOCSTRING_PARSER = DocstringParser
    _DOCSTRING_RENDERER = DocstringRenderer
    __child_func: Callable[..., Any]
    __similarity_ratio: ClassVar[float] = get_similarity_ratio(
        os.environ.get("DOCSTRING_INHERITANCE_SIMILARITY_RATIO")
    )

    def __init__(
        self,
        child_func: Callable[..., Any],
        child_func_signature_updated: bool,
    ) -> None:
        """The __child_func is private. So we need to take the __child_func in the constructor since it is not
        accessible in the parent class."""
        super().__init__(child_func)
        self.__child_func = child_func
        self.__child_func_signature_updated = child_func_signature_updated

    @classmethod
    def inherit(
        cls,
        parent_doc: str | None,
        child_func: Callable[..., Any],
        parent_func: Callable[..., Any] | None = None,
        child_func_signature_updated: bool = False,
    ) -> None:
        """
        Args:
            parent_doc: The docstring of the parent.
            child_func: The child function which docstring inherit from the parent.
        """  # noqa: D205, D212
        if parent_doc is not None:
            cls(child_func, child_func_signature_updated)._inherit(parent_doc, parent_func)

    def _inherit(
        self,
        parent_doc: str,
        parent_func: Callable[..., Any] | None = None,
    ) -> None:
        """Inherit the docstrings of a class.

        Args:
            parent_doc: The docstring of the parent.
        """
        parse = self._DOCSTRING_PARSER.parse
        parent_sections = parse(parent_doc)
        child_sections = parse(self.__child_func.__doc__)

        # Try to load the processed __griffe_signature__ if exists
        parent_func_has_signature = parent_func is not None and hasattr(
            parent_func, "__signature__"
        )
        parent_func_original_signature = getattr(parent_func, "__signature__", None)
        child_func_has_signature = hasattr(self.__child_func, "__signature__")
        child_func_original_signature = getattr(self.__child_func, "__signature__", None)
        if hasattr(parent_func, "__griffe_signature__"):
            parent_func.__signature__ = parent_func.__griffe_signature__
        if hasattr(self.__child_func, "__griffe_signature__"):
            self.__child_func.__signature__ = self.__child_func.__griffe_signature__

        self._warn_similar_sections(parent_sections, child_sections)
        self._inherit_sections(
            parent_sections,
            child_sections,
        )

        # Get the original function eventually behind decorators.
        unwrap(self.__child_func).__doc__ = self._DOCSTRING_RENDERER.render(child_sections)

        # Inherit the annotations and defaults from the parent function.
        # the docstring with the inherited parameters.
        self._inherit_signature(parent_func, child_sections)

        # Restore the original signatures if they were modified.
        if hasattr(parent_func, "__griffe_signature__"):
            if parent_func_has_signature:
                parent_func.__signature__ = parent_func_original_signature
            else:
                # Only delete if the attribute actually exists
                if hasattr(parent_func, "__signature__"):
                    delattr(parent_func, "__signature__")
        if hasattr(self.__child_func, "__griffe_signature__"):
            if child_func_has_signature:
                self.__child_func.__signature__ = child_func_original_signature
            else:
                # Only delete if the attribute actually exists
                if hasattr(self.__child_func, "__signature__"):
                    delattr(self.__child_func, "__signature__")

    def _param_should_skip(self, param: inspect.Parameter) -> bool:
        """Check if the parameter should be skipped."""
        # Skip 'self' and 'cls' parameters, as they are not relevant for kwargs docstrings.
        return param.name in ("self", "cls") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )

    def _inherit_signature(
        self,
        parent_func: Callable[..., Any] | None,
        child_sections: SectionsType,
    ) -> None:
        """Inherit the annotations and defaults from the parent function based on the merged args from `_inherit`. It
        caches the signature in the child function as `__griffe_signature__` to avoid any possible conflicts with
        existing signatures."""
        if self.__child_func_signature_updated or parent_func is None:
            # If no parent function is provided or the signature is updated before, we cannot inherit annotations and
            # defaults.
            return

        full_arg_spec = getfullargspec(self.__child_func)
        if full_arg_spec.varkw is not None:
            child_sig = inspect.signature(self.__child_func)
            parent_sig = inspect.signature(parent_func)

            # Args in the child function
            args_section = child_sections.get(self._DOCSTRING_PARSER.ARGS_SECTION_NAME, {})

            # Check all arguments that not in the child function but in the parent function.
            child_params = []

            # First, add all non-kwargs parameters from child
            for name, param in child_sig.parameters.items():
                if not self._param_should_skip(param):
                    child_params.append(param)

            # Then, add parent parameters that are documented but not in child signature
            # This should include all parameters from parent that appear in docstring
            for arg in args_section:
                if arg not in child_sig.parameters and arg in parent_sig.parameters:
                    # Add the parameter with its original annotation and default
                    parent_param = parent_sig.parameters[arg]
                    child_params.append(parent_param)

            # Also add any parent parameters that might not be documented yet
            # but could be missing (more aggressive approach)
            for param_name, param in parent_sig.parameters.items():
                if (
                    param_name not in child_sig.parameters
                    and not self._param_should_skip(param)
                    and not any(p.name == param_name for p in child_params)
                ):
                    child_params.append(param)

            # Don't add the **kwargs parameter from the child function since all parameters have been inherited
            # from the parent function. This is to avoid duplication of **kwargs in the child function shown
            # in the docs which makes people confused.
            # for name, param in child_sig.parameters.items():
            #     if param.kind == inspect.Parameter.VAR_KEYWORD:
            #         child_params.append(param)

            new_child_sig = inspect.Signature(parameters=child_params)

            # Set both __signature__ and try to update __annotations__ as well
            self.__child_func.__griffe_signature__ = new_child_sig

            # Also update __annotations__ to include the new parameters
            if not hasattr(self.__child_func, "__annotations__"):
                self.__child_func.__annotations__ = {}

            for param in child_params:
                if param.annotation != inspect.Parameter.empty:
                    self.__child_func.__annotations__[param.name] = param.annotation

            # Also try to update the argspec for older inspection methods
            # try:
            #     # Force update of cached inspection data
            #     if hasattr(inspect, "_signature_cache"):
            #         inspect._signature_cache.pop(self.__child_func, None)

            #     # Update the function's __code__ object to include new parameter names
            #     # This is a more aggressive approach for stubborn introspection tools
            #     original_code = self.__child_func.__code__
            #     new_varnames = (
            #         tuple(
            #             param.name
            #             for param in child_params
            #             if param.kind
            #             in (
            #                 inspect.Parameter.POSITIONAL_ONLY,
            #                 inspect.Parameter.POSITIONAL_OR_KEYWORD,
            #                 inspect.Parameter.KEYWORD_ONLY,
            #             )
            #         )
            #         + original_code.co_varnames[original_code.co_argcount :]
            #     )

            #     # Update argcount to include the new parameters
            #     new_argcount = len(
            #         [
            #             p
            #             for p in child_params
            #             if p.kind
            #             in (
            #                 inspect.Parameter.POSITIONAL_ONLY,
            #                 inspect.Parameter.POSITIONAL_OR_KEYWORD,
            #             )
            #         ]
            #     )

            #     # Create new code object with updated parameter information
            #     if hasattr(original_code, "replace"):  # Python 3.8+
            #         new_code = original_code.replace(
            #             co_varnames=new_varnames, co_argcount=new_argcount
            #         )
            #         self.__child_func.__code__ = new_code

            # except Exception:
            #     # If code object modification fails, that's okay - signature should still work
            #     pass

    def _filter_args_section(
        self,
        missing_arg_text: str,
        section_items: dict[str, str],
        section_name: str = "",
    ) -> dict[str, str]:
        """Filter the args section items with the args of a signature.

        The argument ``self`` is removed. The arguments are ordered according to the
        signature of ``func``. An argument of ``func`` missing in ``section_items`` gets
        a default description defined in :attr:`._MISSING_ARG_TEXT`.

        Args:
            missing_arg_text: This text for the missing arguments.
            section_name: The name of the section.
            section_items: The docstring section items.

        Returns:
            The section items filtered with the function signature.
        """
        # Some functions need to be run this function again, such as the __init__ method of a class if the docstring
        # is defined in the class docstring. The __init__ method already has a __griffe_signature__, so we need to
        # parse this one.
        full_arg_spec = getfullargspec(self.__child_func)

        all_args = full_arg_spec.args
        if "self" in all_args:
            all_args.remove("self")

        if full_arg_spec.varargs is not None:
            all_args += [f"*{full_arg_spec.varargs}"]

        all_args += full_arg_spec.kwonlyargs

        if full_arg_spec.varkw is not None:
            all_args += [f"**{full_arg_spec.varkw}"]

        ordered_section = {}
        for arg in all_args:
            if arg in section_items:
                doc = section_items[arg]
            else:
                doc = missing_arg_text
                self._warn(section_name, f"the docstring for the argument '{arg}' is missing.")
            ordered_section[arg] = doc

        if not self.__child_func_signature_updated:
            # If the function has kwargs, it would inherit all other arguments in the
            # parent docstring that haven't been included in the child function arguments.
            full_arg_spec = getfullargspec(self.__child_func)
            if full_arg_spec.varkw is not None:
                for arg, doc in section_items.items():
                    if arg not in ordered_section:
                        ordered_section[arg] = doc

                # Pop *args, **kwargs (if present) from the ordered_section dict
                varargs = f"*{full_arg_spec.varargs}"
                ordered_section.pop(varargs, None)
                varkw = f"**{full_arg_spec.varkw}"
                ordered_section.pop(varkw, None)

        return ordered_section


class GoogleKwargsDocstringInheritanceInitMeta(_BaseDocstringInheritanceMeta):
    """Metaclass for inheriting docstrings in Google format with init-in-class. This class is used to inherit all args
    in the parent docstring if the child docstring has kwargs."""

    def __init__(
        cls,
        class_name: str,
        class_bases: tuple[type],
        class_dict: dict[str, Any],
    ) -> None:
        super().__init__(
            class_name,
            class_bases,
            class_dict,
            GoogleKwargsDocstringInheritor,
            init_in_class=True,
        )
