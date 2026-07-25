from graph.graph import app

def main():
    response = app.invoke({"question": "HOW TO MAKE PIZZA"})
    print(response["generation"].answer)


if __name__ == "__main__":
    main()
