"""
Removes duplicate sentences
"""


def get_before_comma(line):
    i = line.find('.')
    return line[:i + 2]


x = '''
"The pitcher threw the ball.", "The pitcher threw the ball."
"The ball crossed the plate outside of the strike zone.", "The ball crossed the plate outside of the strike zone. The ball crossed the plate."
"The batter hit the ball.", "The batter hit the ball."
"The ball did not stay in the field of play.", "The ball did not stay in the field."
"The umpire ruled that the batter did not swing.", "The batter did not swing."
"It was a ball.", "It was a ball."
"The first base umpire declared that the batter did not take a full swing.", "The batter did not take a full swing."
"The pitch was called a ball.", "The pitch was called a ball."
"The ball hit the ground before it was caught.", "The ball hit the ground before it was caught. The ball hit the ground."
"The batted ball landed in shallow center field.", "The batted ball landed in shallow center field. The batted ball landed."
"The batted ball landed for a hit.", "The batted ball landed for a hit. The batted ball landed." 
"The pitch crossed the plate inside of the strike zone.", "The pitch crossed the plate inside of the strike zone. The pitch crossed the plate."
"The pitch was called a strike by the umpire.", "The pitch was called a strike."
"The hitter hit the ball straight up the middle to the pitcher.", "The hitter hit the ball straight up the middle to the pitcher. The hitter hit the ball straight up the middle. The hitter hit the ball."
"The pitcher caught it.", "The pitcher caught it."
"The pitcher threw to 2nd base.", "The pitcher threw to 2nd base. The pitcher threw."
"The pitcher got the runner out.", "The pitcher got the runner out."
"The hitter hit the ball straight up the middle.", "The hitter hit the ball straight up the middle. The hitter hit the ball."
"The hitter hit the ball to the pitcher.", "The hitter hit the ball to the pitcher. The hitter hit the ball."
"The pitcher got the runner out.", "The pitcher got the runner out."
"The 2nd baseman threw to first baseman.", "The 2nd baseman threw to first baseman. The 2nd baseman threw"
"The 2nd baseman got the hitter out.", "The 2nd baseman got the hitter out."
"The umpire behind the plate appealed to the first base umpire.", "The umpire behind the plate appealed to the first base umpire. The umpire behind the plate appealed. The umpire appealed to the umpire. The umpire appealed."
"The first base umpire had a better view.", "The first base umpire had a better view."
"The first base umpire declared that the batter did not take a full swing.", "The first base umpire declared that the batter did not take a full swing. The batter did not take a full swing."
"The pitch was called a ball, rather than a strike.", "The pitch was called a ball, rather than a strike. The pitch was called a ball."
'''

if __name__ == "__main__":
    print("Hello world")

    s = set()

    with open('new_concept_generalisation_examples.csv') as f:
        seen = False
        for line in f:
            sent = get_before_comma(line)
            s.add(sent)

    for line in x.splitlines():
        if line != '':
            solved = get_before_comma(line)
            if solved in s:
                s.remove(solved)

    for elem in s:
        print(elem)
