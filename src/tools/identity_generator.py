"""IdentityGenerator – generates synthetic identity data for web form filling.

Provides a single public function ``generate_identity()`` that returns a dict
containing a consistent set of random but realistic-looking personal data
(name, email, password, birthdate, phone number, username) suitable for
web form auto-fill during registration / sign-up tasks.

The generated data uses only Python's standard library (``random``, ``string``)
so it has no external dependencies.
"""

from __future__ import annotations

import random
import string

_FIRST_NAMES = [
    "Andi", "Budi", "Citra", "Dewi", "Eko", "Fitri", "Gilang", "Hana",
    "Irfan", "Jasmine", "Kevin", "Lina", "Muhammad", "Nadia", "Oscar",
    "Putri", "Rizky", "Sari", "Teguh", "Ulfa", "Vera", "Wahyu", "Yeni", "Zaki",
    "Agus", "Bagas", "Cahya", "Dimas", "Eka", "Farah", "Gita", "Hendra",
    "Indra", "Joko", "Kurnia", "Layla", "Marwa", "Nisa", "Okta", "Reza",
]

_LAST_NAMES = [
    "Pratama", "Santoso", "Rahayu", "Purnama", "Wijaya", "Kusuma", "Sari",
    "Hidayat", "Permata", "Setiawan", "Nugroho", "Gunawan", "Hasan",
    "Irawati", "Juniarsa", "Kartika", "Lubis", "Maulana", "Nasution",
    "Oktaviani", "Prasetyo", "Qomariah", "Ramadhan", "Suryadi",
    "Tambunan", "Utami", "Valentina", "Wibowo", "Yusuf", "Zainudin",
]

_EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]

_SYMBOLS = "!@#$%^&*"


def generate_identity(seed: int | None = None) -> dict[str, str]:
    """Generate a set of random but realistic-looking identity fields.

    All fields are generated from a single RNG instance so the result is
    internally consistent (the username and email both derive from the same
    first/last name).

    Args:
        seed: Optional integer RNG seed for reproducible output in tests.
              When ``None`` (the default) the system random source is used.

    Returns:
        A ``dict`` with the following string keys:

        ``first_name``  – e.g. ``"Andi"``
        ``last_name``   – e.g. ``"Pratama"``
        ``full_name``   – e.g. ``"Andi Pratama"``
        ``username``    – e.g. ``"andi_pratama42"``
        ``email``       – e.g. ``"andi.pratama9472@gmail.com"``
        ``password``    – ≥10 chars, mixed upper/lower/digit/symbol
        ``birthdate``   – ``"DD/MM/YYYY"`` format, age 23–37 in 2025
        ``phone``       – Indonesian mobile format ``"08xxxxxxxxx"``
    """
    rng = random.Random(seed)

    first  = rng.choice(_FIRST_NAMES)
    last   = rng.choice(_LAST_NAMES)
    num    = rng.randint(10, 9999)
    domain = rng.choice(_EMAIL_DOMAINS)

    # Email: lower-case name + random digits + domain
    email = f"{first.lower()}.{last.lower()}{num}@{domain}"

    # Username: underscore-joined lower-case name + 2-digit suffix
    username = f"{first.lower()}_{last.lower()}{rng.randint(10, 99)}"

    # Password: ≥10 chars, guaranteed 1 uppercase, 1 lowercase, 1 digit, 1 symbol
    pwd_chars = (
        rng.choices(string.ascii_uppercase, k=2)
        + rng.choices(string.ascii_lowercase, k=4)
        + rng.choices(string.digits,          k=2)
        + rng.choices(_SYMBOLS,               k=2)
    )
    rng.shuffle(pwd_chars)
    password = "".join(pwd_chars)

    # Birthdate: random date between 1988 and 2002 (age 23–37 in 2025)
    birth_year  = rng.randint(1988, 2002)
    birth_month = rng.randint(1, 12)
    birth_day   = rng.randint(1, 28)  # safe for all months
    birthdate   = f"{birth_day:02d}/{birth_month:02d}/{birth_year}"

    # Indonesian mobile: "08" + 9 random digits (11 digits total)
    phone = "08" + "".join(rng.choices(string.digits, k=9))

    return {
        "first_name": first,
        "last_name":  last,
        "full_name":  f"{first} {last}",
        "username":   username,
        "email":      email,
        "password":   password,
        "birthdate":  birthdate,
        "phone":      phone,
    }
