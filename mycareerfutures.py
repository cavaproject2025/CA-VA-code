import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import json

def get_salary_info(job_role, headless=True, timeout=20):
    """
    Scrapes MyCareersFuture website for salary information for a given job role.

    Args:
        job_role (str): The job role to search for (e.g., 'Data Scientist', 'Software Engineer')
        headless (bool): Whether to run the browser in headless mode (no UI)
        timeout (int): Maximum time to wait for elements to load in seconds

    Returns:
        dict: Dictionary containing salary information for the job role
    """
    # Set up Selenium webdriver with Chrome
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")  # Using newer headless mode
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=chrome_options)

    try:
        # Navigate to MyCareersFuture
        driver.get("https://www.mycareersfuture.gov.sg/")
        print(f"Searching for salary information for: {job_role}")

        # Wait for page to load completely
        time.sleep(3)

        # Print page title for debugging
        print(f"Page title: {driver.title}")

        # Try multiple selectors for the search box
        search_box = None
        selectors = [
            "input[placeholder='Search for keywords']",
            "input[type='text']",
            "input.search-input",
            "//input[@placeholder='Search for keywords']",
            "//input[contains(@class, 'search')]"
        ]

        for selector in selectors:
            try:
                if selector.startswith("//"):
                    search_box = WebDriverWait(driver, timeout).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                else:
                    search_box = WebDriverWait(driver, timeout).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                print(f"Found search box using selector: {selector}")
                break
            except Exception as e:
                print(f"Selector {selector} failed: {e.__class__.__name__}")
                continue

        if not search_box:
            # Take screenshot for debugging
            driver.save_screenshot("debug_screenshot.png")
            print("Could not find search box. Screenshot saved as debug_screenshot.png")
            print("HTML source snippet:")
            print(driver.page_source[:500] + "...")
            return {"error": "Could not find search input on the website"}

        # Continue with the search
        search_box.clear()
        search_box.send_keys(job_role)
        search_box.send_keys(Keys.RETURN)

        # Wait for search results to load
        print("Waiting for search results...")
        try:
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR,
                    "[data-testid='job-card'], .search-card, .job-card, article, div[class*='JobCard']"))
            )
            print("Search results loaded")
            # Take screenshot for debugging
            driver.save_screenshot("search_results.png")
            print("Search results screenshot saved")
        except Exception as e:
            print(f"Error waiting for search results: {e}")
            driver.save_screenshot("search_results_debug.png")
            return {"error": f"Could not find search results for {job_role}"}

        # Extract salary information from the search results
        salary_data = {
            "job_role": job_role,
            "salary_range": [],
            "average_salary": None,
            "median_salary": None,
            "sample_size": 0
        }

        # Try to find a salary indicator or insights section
        print("Looking for salary insights...")
        try:
            # Check if there's a salary insights section
            salary_elements = driver.find_elements(By.XPATH,
                "//*[contains(text(), 'Salary')]/ancestor::div[3]")

            if salary_elements:
                print(f"Found {len(salary_elements)} potential salary elements")
                for el in salary_elements:
                    print(f"Salary element text: {el.text[:100]}...")
                    if "$" in el.text:
                        # Extract all salary ranges using regex
                        salary_matches = re.findall(r'\$[\d,]+\s*-\s*\$[\d,]+|\$[\d,]+', el.text)
                        if salary_matches:
                            for match in salary_matches:
                                if match not in salary_data["salary_range"]:
                                    salary_data["salary_range"].append(match)
                            print(f"Extracted salary ranges: {salary_matches}")

                        # Try to find average and median
                        if "average" in el.text.lower():
                            avg_match = re.search(r'average.*?(\$[\d,]+)', el.text.lower())
                            if avg_match:
                                salary_data["average_salary"] = avg_match.group(1)
                                print(f"Found average salary: {avg_match.group(1)}")

                        if "median" in el.text.lower():
                            med_match = re.search(r'median.*?(\$[\d,]+)', el.text.lower())
                            if med_match:
                                salary_data["median_salary"] = med_match.group(1)
                                print(f"Found median salary: {med_match.group(1)}")
        except Exception as e:
            print(f"Error processing salary insights: {e}")

        # If no salary insights found, try job cards
        if not salary_data["salary_range"]:
            print("Looking for salary info in job cards...")

            # Try multiple selectors for job cards
            job_card_selectors = [
                "[data-testid='job-card']",
                ".search-card",
                ".job-card",
                "article",
                "div[class*='JobCard']",
                "div[class*='job-card']",
                "//div[contains(@class, 'job') and contains(@class, 'card')]"
            ]

            job_cards = []
            for selector in job_card_selectors:
                try:
                    if selector.startswith("//"):
                        cards = driver.find_elements(By.XPATH, selector)
                    else:
                        cards = driver.find_elements(By.CSS_SELECTOR, selector)

                    if cards:
                        print(f"Found {len(cards)} job cards using selector: {selector}")
                        job_cards = cards
                        break
                except Exception as e:
                    print(f"Job card selector {selector} failed: {e.__class__.__name__}")

            # Process job cards for salary information
            for card in job_cards[:15]:  # Check first 15 job postings
                try:
                    card_text = card.text
                    print(f"Job card text snippet: {card_text[:50]}...")

                    # Try to extract salary using regex
                    salary_matches = re.findall(r'\$[\d,]+\s*-\s*\$[\d,]+|\$[\d,]+', card_text)
                    if salary_matches:
                        for match in salary_matches:
                            if match not in salary_data["salary_range"]:
                                salary_data["salary_range"].append(match)
                        print(f"Extracted salary ranges: {salary_matches}")
                except Exception as e:
                    print(f"Error processing job card: {e.__class__.__name__}")
                    continue

            if salary_data["salary_range"]:
                salary_data["sample_size"] = len(salary_data["salary_range"])
                print(f"Found {len(salary_data['salary_range'])} job postings with salary information")

        # If still no salary info, try one more approach - scan entire page for salary patterns
        if not salary_data["salary_range"]:
            print("Scanning entire page for salary information...")
            page_text = driver.page_source
            salary_matches = re.findall(r'\$[\d,]+\s*-\s*\$[\d,]+|\$[\d,]+', page_text)
            if salary_matches:
                # Filter out duplicates
                unique_salaries = list(set(salary_matches))
                salary_data["salary_range"] = unique_salaries
                salary_data["sample_size"] = len(unique_salaries)
                print(f"Found {len(unique_salaries)} salary mentions on the page")

        # Calculate average and median from the collected salary ranges
        if salary_data["salary_range"]:
            salary_values = []

            for salary_str in salary_data["salary_range"]:
                # Handle ranges like "$5,000 - $8,000"
                if "-" in salary_str:
                    parts = salary_str.split("-")
                    min_sal = float(parts[0].replace("$", "").replace(",", "").strip())
                    max_sal = float(parts[1].replace("$", "").replace(",", "").strip())
                    # Use the midpoint of the range
                    salary_values.append((min_sal + max_sal) / 2)
                else:
                    # Handle single values like "$5,000"
                    salary_values.append(float(salary_str.replace("$", "").replace(",", "").strip()))

            if salary_values:
                # Calculate average
                avg_salary = sum(salary_values) / len(salary_values)
                salary_data["average_salary"] = f"${int(avg_salary):,}"

                # Calculate median
                sorted_values = sorted(salary_values)
                mid = len(sorted_values) // 2
                if len(sorted_values) % 2 == 0:
                    median_salary = (sorted_values[mid-1] + sorted_values[mid]) / 2
                else:
                    median_salary = sorted_values[mid]
                salary_data["median_salary"] = f"${int(median_salary):,}"

                print(f"Calculated average salary: {salary_data['average_salary']}")
                print(f"Calculated median salary: {salary_data['median_salary']}")

        return salary_data

    except Exception as e:
        print(f"Unexpected error: {e}")
        driver.save_screenshot("error_screenshot.png")
        return {"error": str(e)}
    finally:
        # Clean up
        driver.quit()

# Example usage
if __name__ == "__main__":
    job_role = "Sales Assistant"
    salary_info = get_salary_info(job_role)
    print(f"\nSalary information for {job_role}:")
    print(json.dumps(salary_info, indent=2))