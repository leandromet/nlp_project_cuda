#!/usr/bin/env python3
"""
Test script to verify changes calculation logic
"""

import numpy as np

def test_changes_calculation():
    """Test the changes calculation logic with simple data"""
    print("=== Testing Changes Calculation Logic ===")
    
    # Create simple test data: 3 years, 4x4 pixels
    # Year 1: [1,1,1,1]  Year 2: [1,2,1,1]  Year 3: [1,2,3,1]
    #         [1,1,1,1]          [1,1,1,1]          [1,1,1,1]
    #         [1,1,1,1]          [1,1,1,1]          [1,1,1,1]
    #         [1,1,1,1]          [1,1,1,1]          [1,1,1,1]
    
    year1 = np.ones((4, 4), dtype='uint8')
    year2 = np.ones((4, 4), dtype='uint8')
    year2[0, 1] = 2  # One pixel changes from 1 to 2
    year3 = year2.copy()
    year3[0, 2] = 3  # Another pixel changes from 1 to 3
    
    print("Year 1:")
    print(year1)
    print("Year 2:")
    print(year2)
    print("Year 3:")
    print(year3)
    
    # Test our changes calculation logic
    changes = np.zeros_like(year1, dtype='uint8')
    previous_year = year1.copy()
    
    # Process year 2
    changed_mask = (previous_year != year2)
    changes[changed_mask] += 1
    print(f"\nAfter year 2, changes mask:")
    print(changed_mask.astype(int))
    print(f"Changes array:")
    print(changes)
    previous_year = year2.copy()
    
    # Process year 3  
    changed_mask = (previous_year != year3)
    changes[changed_mask] += 1
    print(f"\nAfter year 3, changes mask:")
    print(changed_mask.astype(int))
    print(f"Final changes array:")
    print(changes)
    
    # Expected: [0,1] should have 1 change, [0,2] should have 1 change
    expected_changes = np.zeros((4, 4), dtype='uint8')
    expected_changes[0, 1] = 1  # Changed in year 2
    expected_changes[0, 2] = 1  # Changed in year 3
    
    print(f"\nExpected changes:")
    print(expected_changes)
    print(f"Matches expected: {np.array_equal(changes, expected_changes)}")
    
    # Test persistence logic too
    persistence = np.ones_like(year1, dtype=bool)
    reference = year1.copy()
    
    # Check year 2
    persistence &= (reference == year2)
    print(f"\nPersistence after year 2:")
    print(persistence.astype(int))
    
    # Check year 3
    persistence &= (reference == year3)
    print(f"Final persistence:")
    print(persistence.astype(int))
    
    # Count persistent vs changed
    persistent_pixels = np.sum(persistence)
    changed_pixels = np.sum(changes > 0)
    total_pixels = changes.size
    
    print(f"\nSummary:")
    print(f"Total pixels: {total_pixels}")
    print(f"Persistent pixels: {persistent_pixels}")
    print(f"Changed pixels: {changed_pixels}")
    print(f"Sum should equal total: {persistent_pixels + changed_pixels} == {total_pixels}")

if __name__ == "__main__":
    test_changes_calculation()
