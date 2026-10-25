def calculate_score(activity, delta_tm, target_temperature, topt):
    activity_score = min(activity, 120) / 120 * 50

    stability_score = min(max(delta_tm + 5, 0), 10) / 10 * 30

    temperature_difference = abs(topt - target_temperature)

    temperature_score = max(0, 20 - temperature_difference) / 20 * 20

    total_score = (
        activity_score
        + stability_score
        + temperature_score
    )

    return round(total_score, 2)