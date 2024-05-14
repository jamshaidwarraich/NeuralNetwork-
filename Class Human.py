class human:
    def __init__(self, name, gender, height):
        self.name = name
        self.gender = gender
        self.height = height
    
    def status(self):
        print("Name of person is", self.name)
        print(f"Gender of {self.name} is", self.gender)
        print(f"Height of {self.name} is", self.height)

person1 = human("Zahid", "male", "152cm")
person2 = human("Ahsin", "male", "150cm")
person3 = human("Manahil", "female", "145cm")
person4 = human("Zanib", "female", "146cm")

person1.status()
print(".......................")
person2.status()
print(".......................")
person3.status()
print(".......................")
person4.status()
print(".......................")
