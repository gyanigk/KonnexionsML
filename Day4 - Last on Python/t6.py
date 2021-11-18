banned_users = ['Gyanig', 'Ron', 'Harry potter', 'Koyal']
prompt = "\nAdd a player to your team."
prompt += "\nEnter 'quit' when you're done. "
players = []

#while loop below , Hi im Gyanig 

while True:
    player = input(prompt)
    if player == 'quit':
        break
    elif player in banned_users:
        print(f"{player} is banned!")
        continue
    else:
        players.append(player)

print("\nYour team:")
for player in players:
    print(player)
