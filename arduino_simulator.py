def send_to_arduino(density):

    #this function/file simulates what the physical Arduino would do after receiving the density
    #the python program

    print("ARDUINO SIMULATION")
    print(f"Received: {density}")


    #LOW = very light crowd -> green LED, no buzzer
    if density == "LOW":

        print("Green LED: ON")
        print("Yellow LED: OFF")
        print("Orange LED: OFF")
        print("Red LED: OFF")
        print("Buzzer: OFF")
        print("LCD: CROWD - LOW")


    #MEDIUM = just around normal -> yellow LED, no buzzer
    elif density == "MEDIUM":
        print("Green LED: OFF")
        print("Yellow LED: ON")
        print("Orange LED: OFF")
        print("Red LED: OFF")
        print("Buzzer: OFF")
        print("LCD: CROWD - MEDIUM")

    #HIGH = warning that density is increaasing -> orange LED + buzzer
    elif density == "HIGH":
        print("Green LED: OFF")
        print("Yellow LED: OFF")
        print("Orange LED: ON")
        print("Red LED: OFF")
        print("Buzzer: ON")
        print("LCD: CROWD - HIGH")

    #CRITICAL = highest level -> red LED + buzzer
    elif density == "CRITICAL":
        print("Green LED: OFF")
        print("Yellow LED: OFF")
        print("Orange LED: OFF")
        print("Red LED: ON")
        print("Buzzer: ON")
        print("LCD: CROWD - CRITICAL")