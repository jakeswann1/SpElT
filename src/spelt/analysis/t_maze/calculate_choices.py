import numpy as np


def calculate_choices(xy_positions, sector_numbers):
    """
    Function to calculate choice statistics based on sector numbers for each XY position

    Parameters:
    xy_positions (numpy.ndarray): An array containing the XY positions.
        This input is not used in the current function version.
    sector_numbers (numpy.ndarray): An array of sector numbers for each XY position.
        Sector numbers should be between 1 and 12 inclusive.

    Returns:
    dict: A dictionary containing the following keys:
        - total_choices: The total number of valid choices made.
        - total_left_choices: The total number of times a choice was made to go left.
        - total_right_choices: The total number of times a choice was made to go right.
        - total_correct_choices: The total number of correct choices made.
            A correct choice is defined as a choice to go in the opposite direction of
            the previous choice.
        - p_correct: The proportion of correct choices made
            (total_correct_choices / total_choices).
        - p_left_choices: The proportion of left choices made
            (total_left_choices / total_choices).

    Raises:
    ValueError: If the sector numbers are not between 1 and 12 inclusive.

    Notes:
    The function assumes that the animal starts in the center sector
    (sector numbers 5, 6, 7, 8). If the animal starts in the left sectors
    (sector numbers 1, 2, 3, 4) or the right sectors (sector numbers 9, 10, 11, 12),
    a message will be printed to the console. If the sector numbers are not within these
    ranges, an error message will be printed.

    The function uses a state variable `choice` to keep track of the current choice
    (left, right, or center), and this is updated based on the current sector number.
    The choice is then encoded as a numeric value (left: 1, right: 2, center: 0) and
    stored in the `arm_ind` array.

    After encoding the choices, the function removes consecutive duplicates to get the
    sequence of arm visits `arm_visit_order`. It then iterates over this sequence to
    count the number of choices made and the number of correct choices. A correct choice
    is defined as a choice to go in the opposite direction of the previous choice.

    Finally, it calculates the proportion of correct choices and the proportion of left
    choices and returns these values in a dictionary.
    """
    arm_ind = np.zeros(len(sector_numbers))

    choice = ""
    # # Initialize choice as starting arm (usually centre)
    # if sector_numbers[0] in [5,6,7,8]:
    #     choice = 'centre'
    # elif sector_numbers[0] in [1,2,3,4]:
    #     choice = 'left'
    #     print('Animal not starting trial in centre arm')
    # elif sector_numbers[0] in [9, 10, 11, 12]:
    #     choice = 'right'
    #     print('Animal not starting trial in centre arm')
    # else:
    #     choice = ''
    #     print('Sectors assigned incorrectly, please check')

    for i in range(len(sector_numbers)):
        if sector_numbers[i] == 8:
            choice = "centre"
        elif sector_numbers[i] == 1:
            choice = "left"
        elif sector_numbers[i] == 9:
            choice = "right"

        if choice == "centre":
            arm_ind[i] = 0
        elif choice == "left":
            arm_ind[i] = 1
        elif choice == "right":
            arm_ind[i] = 2
        else:
            arm_ind[i] = -1

    # Calculate total choice counts & proportion correct
    # Create sequence of arm visits
    arm_visit_order = arm_ind[np.concatenate([[True], np.diff(arm_ind) != 0])]

    # Initialize counters
    total_choices = 0
    total_left_choices = 0
    total_right_choices = 0
    total_correct_choices = 0

    # Iterate over the arm_visit_order array
    previous_choice = -1  # Initialize previous choice to an invalid value
    for i in range(len(arm_visit_order)):
        current_choice = arm_visit_order[i]

        if current_choice == 0:
            # Central arm visit
            if previous_choice != -1:
                # Check if the previous choice was valid
                next_value_indices = np.where(arm_visit_order[i + 1 :] != 0)[0]
                if len(next_value_indices) > 0:
                    next_value = arm_visit_order[i + 1 + next_value_indices[0]]

                    if next_value == 1:
                        total_left_choices += 1
                    elif next_value == 2:
                        total_right_choices += 1

                    # Increment the total number of choices
                    total_choices += 1

                    if (previous_choice == 1 and next_value == 2) or (
                        previous_choice == 2 and next_value == 1
                    ):
                        # Correct choice (opposite arm to the previous choice)
                        total_correct_choices += 1
        # Update the previous choice
        previous_choice = current_choice

    try:
        # Calculate the proportion of correct choices
        p_correct = total_correct_choices / total_choices

        # Calculate the proportion of left choices
        p_left_choices = total_left_choices / total_choices

    # Catch case where n_choices = 0
    except ZeroDivisionError:
        p_correct = np.nan
        p_left_choices = np.nan

    return {
        "total_choices": total_choices,
        "total_left_choices": total_left_choices,
        "total_right_choices": total_right_choices,
        "total_correct_choices": total_correct_choices,
        "p_correct": p_correct,
        "p_left_choices": p_left_choices,
    }
