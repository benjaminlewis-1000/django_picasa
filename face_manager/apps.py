from django.apps import AppConfig
from django.core.checks import Error, register


class FaceManagerConfig(AppConfig):
    name = 'face_manager'

    def ready(self):
        register(check_num_possible_identities)


def check_num_possible_identities(app_configs, **kwargs):
    # Face.NUM_POSSIBLE_IDENTITIES is the single source of truth for how
    # many poss_identN/weight_N field pairs the model has (and how many
    # remove_poss_ident() et al. loop over). If someone adds/removes a
    # field pair without updating the constant -- or vice versa -- code
    # relying on the constant would silently skip or miss fields instead
    # of failing. This check makes that drift a loud manage.py check error.
    from face_manager.models import Face

    errors = []
    field_names = {f.name for f in Face._meta.get_fields()}
    expected_n = Face.NUM_POSSIBLE_IDENTITIES

    actual_indices = set()
    for name in field_names:
        for prefix in ('poss_ident', 'weight_'):
            if name.startswith(prefix) and name[len(prefix):].isdigit():
                actual_indices.add(int(name[len(prefix):]))

    expected_indices = set(range(1, expected_n + 1))
    if actual_indices != expected_indices:
        errors.append(
            Error(
                f"Face.NUM_POSSIBLE_IDENTITIES is {expected_n} (expects "
                f"poss_identN/weight_N fields for N in {sorted(expected_indices)}), "
                f"but the model actually defines fields for N in {sorted(actual_indices)}. "
                "Update NUM_POSSIBLE_IDENTITIES to match, or add/remove the "
                "missing/extra field pairs.",
                id='face_manager.E001',
            )
        )
    return errors
