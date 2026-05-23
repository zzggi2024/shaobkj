# Hack: string type that is always equal in not equal comparisons
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")


class Shaobkj_ParamExtract:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"value": (any_type,)},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("value",)
    FUNCTION = "route"
    CATEGORY = "🤖shaobkj-APlbox"

    def route(self, value):
        return (value,)
