from __future__ import annotations


def patch_cosysairsim_msgpack(airsim_module) -> None:
    types_module = getattr(airsim_module, "types", None)
    if types_module is None:
        import cosysairsim.types as types_module

    mixin_cls = types_module.MsgpackMixin
    if getattr(mixin_cls, "_uavmissionplanning_patched", False):
        return

    @classmethod
    def _from_msgpack(cls, encoded):
        obj = cls()
        if not isinstance(encoded, dict):
            return encoded

        for raw_key, value in encoded.items():
            key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else raw_key
            if not isinstance(key, str):
                continue

            try:
                current_value = getattr(obj, key)
            except AttributeError:
                obj.__dict__[key] = value
                continue

            if isinstance(value, dict) and hasattr(current_value.__class__, "from_msgpack"):
                obj.__dict__[key] = current_value.__class__.from_msgpack(value)
            else:
                obj.__dict__[key] = value

        return obj

    mixin_cls.from_msgpack = _from_msgpack
    mixin_cls._uavmissionplanning_patched = True