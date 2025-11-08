"""Griffe extension for kwargs docstring inheritance."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any
from typing import ClassVar
from typing import Literal
from griffe import (
    Alias,
    Attribute,
    Class,
    Extension,
    Inspector,
    Object,
    ObjectNode,
    Visitor,
    dynamic_import,
    get_logger,
    Docstring,
)

if TYPE_CHECKING:
    import ast

    from griffe import Parser


_logger = get_logger(__name__)


class KwargsDocstringInheritance(Extension):
    """Handle kwargs-based parameter inheritance for griffe documentation generation. This class maily inspired by the
    `DocstringInheritance` in `docstring_inheritance/griffe.py`. Here, signatures of methods are updated to inherit
    parameters from parent classes when `GoogleKwargsDocstringInheritanceMeta` is used."""

    __parser: Literal["google", "numpy", "sphinx"] | Parser | None = None
    """The docstring parser."""

    __parser_options: ClassVar[dict[str, Any]] = {}
    """The docstring parser options."""

    def on_class_members(
        self,
        *,
        node: ast.AST | ObjectNode,
        cls: Class,
        agent: Visitor | Inspector,
        **kwargs: Any,
    ) -> None:
        """Process class members to inherit kwargs parameters."""
        _logger.debug("GRIFFE EXTENSION CALLED FOR: %s at %s", cls.name, cls.filepath)

        if isinstance(node, ObjectNode):
            # Skip runtime objects, their docstrings are already OK.
            _logger.debug("  Skipping runtime object: %s", cls.name)
            return

        runtime_cls = self._import_dynamically(cls)

        if not self.__has_docstring_inheritance(runtime_cls):
            _logger.debug("  No docstring inheritance detected for: %s", cls.name)
            return

        # Inherit the class docstring.
        self.__set_docstring(cls, runtime_cls)

        # Inherit the methods docstrings.
        for member in cls.members.values():
            if not isinstance(member, Attribute):
                runtime_obj = self._import_dynamically(member)
                self.__set_docstring(member, runtime_obj)

        # Inherit the kwargs parameters if applicable.
        if not self._has_kwargs_inheritance(runtime_cls):
            _logger.debug("  No kwargs inheritance detected for: %s", cls.name)
            return

        _logger.debug("  FOUND kwargs inheritance for: %s", cls.name)

        # Process __init__ method specifically
        init_method = cls.members.get("__init__")
        _logger.debug("  Init method found: %s", init_method)
        _logger.debug("  Init method type: %s", type(init_method))

        if init_method:
            runtime_init = getattr(runtime_cls, "__init__", None)
            _logger.debug("  Runtime init: %s", runtime_init)
            if runtime_init:
                _logger.debug("  Processing __init__ method for: %s", cls.name)
                self._inherit_kwargs_parameters(init_method, runtime_init, runtime_cls)
            else:
                _logger.debug("  No runtime __init__ found for: %s", cls.name)
        else:
            _logger.debug("  Init method not found for: %s", cls.name)

    @staticmethod
    def _import_dynamically(obj: Object | Alias) -> Any:
        """Import dynamically and return an object."""
        try:
            return dynamic_import(obj.path)
        except ImportError:
            _logger.debug("Could not get dynamic docstring for %s", obj.path)
            return None

    @staticmethod
    def _has_kwargs_inheritance(cls: type[Any]) -> bool:
        """Check if a class uses kwargs-based docstring inheritance."""
        if cls is None:
            return False

        # Check the metaclass and base classes for kwargs inheritance
        for base in cls.__class__.__mro__:
            if "GoogleKwargsDocstringInheritance" in base.__name__:
                return True

        # Also check if __init__ method has been modified by our inheritance
        init_method = getattr(cls, "__init__", None)
        if init_method and hasattr(init_method, "__signature__"):
            # If it has a custom signature, likely our inheritance was applied
            return True

        return False

    def _inherit_kwargs_parameters(
        self, griffe_method: Attribute, runtime_method: Any, runtime_cls: type
    ) -> None:
        """Inherit parameters from parent classes when **kwargs is present."""
        try:
            print(f"    _inherit_kwargs_parameters called for: {griffe_method.name}")
            print(f"    griffe_method type: {type(griffe_method)}")

            # Get the signature from the runtime method
            if hasattr(runtime_method, "__griffe_signature__"):
                runtime_sig = runtime_method.__griffe_signature__
            else:
                runtime_sig = inspect.signature(runtime_method)

            print(f"    Runtime signature parameters: {list(runtime_sig.parameters.keys())[:10]}")

            # There is no need to check if there's a **kwargs parameter since all classes from
            # GoogleKwargsDocstringInheritanceInitMeta should inherit arguments from their parent classes.
            # has_kwargs = any(
            #     param.kind == inspect.Parameter.VAR_KEYWORD
            #     for param in runtime_sig.parameters.values()
            # )

            # if not has_kwargs:
            #     print(f"    No **kwargs found in {griffe_method.name}")
            #     return

            # print("    Found **kwargs, updating griffe parameters...")

            # Handle different griffe object types
            target_func = None

            # If griffe_method is a Function directly
            if hasattr(griffe_method, "parameters"):
                target_func = griffe_method
                print("    griffe_method is Function-like, using directly")
            # If griffe_method is an Attribute containing a function
            elif hasattr(griffe_method, "function") and griffe_method.function:
                target_func = griffe_method.function
                print("    griffe_method has function attribute")
            # If griffe_method has a value that's function-like
            elif hasattr(griffe_method, "value") and hasattr(griffe_method.value, "parameters"):
                target_func = griffe_method.value
                print("    griffe_method has function-like value")

            if target_func and hasattr(target_func, "parameters"):
                print(f"    target_func type: {type(target_func)}")
                print(f"    Before: {[p.name for p in target_func.parameters]}")

                # Clear existing parameters and add the runtime ones
                # Parameters object doesn't support clear(), so we need to recreate it
                from griffe import Parameters

                new_parameters = Parameters()

                for param_name, param in runtime_sig.parameters.items():
                    if param_name in ("self", "cls") or param.kind == inspect.Parameter.VAR_KEYWORD:
                        continue

                    # Create griffe parameter
                    griffe_param = self._create_griffe_parameter(param, param_name)
                    new_parameters.add(griffe_param)

                # Replace the parameters collection
                target_func.parameters = new_parameters

                print(f"    After: {[p.name for p in target_func.parameters]}")
            else:
                print("    Could not find target function in griffe_method")
                print(f"    griffe_method attributes: {dir(griffe_method)}")

        except Exception as e:
            print(f"    ERROR in _inherit_kwargs_parameters: {e}")
            import traceback

            traceback.print_exc()
            _logger.debug("Failed to inherit kwargs parameters for %s: %s", griffe_method.path, e)

    def _create_griffe_parameter(self, inspect_param: inspect.Parameter, name: str):
        """Create a griffe parameter from an inspect parameter."""
        from griffe import Parameter, ParameterKind

        # Map inspect kinds to griffe kinds
        kind_mapping = {
            inspect.Parameter.POSITIONAL_ONLY: ParameterKind.positional_only,
            inspect.Parameter.POSITIONAL_OR_KEYWORD: ParameterKind.positional_or_keyword,
            inspect.Parameter.VAR_POSITIONAL: ParameterKind.var_positional,
            inspect.Parameter.KEYWORD_ONLY: ParameterKind.keyword_only,
            inspect.Parameter.VAR_KEYWORD: ParameterKind.var_keyword,
        }

        # Create the griffe parameter
        griffe_param = Parameter(
            name=name,
            kind=kind_mapping.get(inspect_param.kind, ParameterKind.positional_or_keyword),
            annotation=self._get_annotation_string(inspect_param.annotation),
            default=self._get_default_string(inspect_param.default),
        )

        return griffe_param

    def _get_annotation_string(self, annotation) -> str | None:
        """Convert annotation to string representation."""
        if annotation == inspect.Parameter.empty:
            return None

        if hasattr(annotation, "__name__"):
            return annotation.__name__
        elif hasattr(annotation, "__module__") and hasattr(annotation, "__qualname__"):
            return f"{annotation.__module__}.{annotation.__qualname__}"
        else:
            return str(annotation)

    def _get_default_string(self, default) -> str | None:
        """Convert default value to string representation."""
        if default == inspect.Parameter.empty:
            return None

        if isinstance(default, str):
            return repr(default)
        else:
            return str(default)

    @classmethod
    def __set_docstring(cls, obj: Object | Alias, runtime_obj: Any) -> None:
        """Set the docstring from a runtime object.

        Args:
            obj: The griffe object.
            runtime_obj: The runtime object.
        """
        if runtime_obj is None:
            return

        try:
            docstring = runtime_obj.__doc__
        except AttributeError:
            _logger.debug("Object %s does not have a __doc__ attribute", obj.path)
            return

        if docstring is None:
            return

        # Update the object instance with the evaluated docstring.
        if obj.docstring:
            obj.docstring.value = inspect.cleandoc(docstring)
        else:
            assert not isinstance(obj, Alias)
            cls.__find_parser(obj)
            obj.docstring = Docstring(
                docstring,
                parent=obj,
                parser=cls.__parser,
                parser_options=cls.__parser_options,
            )

    @staticmethod
    def __has_docstring_inheritance(cls: type[Any]) -> bool:
        """Return whether a class has docstring inheritance."""
        for base in cls.__class__.__mro__:
            if base.__name__.endswith("DocstringInheritanceMeta"):
                return True

        return False

    @classmethod
    def __find_parser(cls, obj: Object) -> None:
        """Search a docstring parser recursively from an object parents."""
        if cls.__parser is not None:
            return

        parent = obj.parent
        if parent is None:
            msg = f"Cannot find a parent of the object {obj}"
            raise RuntimeError(msg)

        if parent.docstring is None:
            msg = f"Cannot find a docstring for the parent of the object {obj}"
            raise RuntimeError(msg)

        parser = parent.docstring.parser

        if parser is None:
            cls.__find_parser(parent)
        else:
            cls.__parser = parser
            cls.__parser_options = parent.docstring.parser_options
