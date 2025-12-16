import { atom } from 'nanostores';

// Function to get the current time in a 24-hour format (float)
const getInitialTime = () => {
    const now = new Date();
    return now.getHours() + now.getMinutes() / 60;
};

/**
 * A Nanostore for managing the simulated time of day.
 * Value is a number representing the hour (0 to 24).
 * We use 0-24 to allow for smooth transitions between 23:xx and 00:xx.
 */
export const timeStore = atom<number>(getInitialTime());

/**
 * Updates the time store, ensuring the hour loops correctly between 0 and 24.
 * @param delta_hours The change in hours (can be fractional).
 */
export function changeTime(delta_hours: number) {
    const currentHour = timeStore.get();
    let newHour = currentHour + delta_hours;
    
    // Loop time between 0 and 24
    if (newHour >= 24) {
        newHour -= 24;
    } else if (newHour < 0) {
        newHour += 24;
    }

    timeStore.set(newHour);
}
