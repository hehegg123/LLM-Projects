from typing import TypedDict

# Define a TypedDict for user profile data
class UserProfile(TypedDict):
    name: str
    age: int
    is_active: bool

Jing = UserProfile("Jing", "29", 1)

print(Jing)