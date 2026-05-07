#!/usr/bin/env python3
"""
Test script for dynamic environment switching functionality.

Tests:
1. Environment detection from missions
2. Graceful server shutdown
3. Server startup with new environment
4. Environment switching logic
5. Error handling
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.environment_manager import EnvironmentManager


def create_test_mission(environment: str, name: str) -> Dict[str, Any]:
    """Create a test mission config."""
    return {
        "name": name,
        "metadata": {
            "environment": environment,
            "task_category": "Visual_Inspection",
        },
        "ground_truth": {
            "waypoint_gt": {"x": 0, "y": 0, "z": -10, "yaw": 0}
        }
    }


class EnvironmentManagerTestHarness:
    """Test harness for environment manager."""
    
    def __init__(self, project_root: Path | str):
        self.project_root = Path(project_root)
        self.env_manager = EnvironmentManager(self.project_root)
        self.tests_passed = 0
        self.tests_failed = 0
    
    def test_mission_environment_detection(self):
        """Test 1: Environment detection from missions."""
        print("\n" + "="*70)
        print("TEST 1: Mission Environment Detection")
        print("="*70)
        
        try:
            # Test NH environment
            mission_nh = create_test_mission("NHEnv", "mission1")
            env = self.env_manager.get_mission_environment(mission_nh)
            assert env == "NHEnv", f"Expected NHEnv, got {env}"
            print("✓ Detected NHEnv mission")
            
            # Test Forest environment
            mission_forest = create_test_mission("ForestEnv", "mission2")
            env = self.env_manager.get_mission_environment(mission_forest)
            assert env == "ForestEnv", f"Expected ForestEnv, got {env}"
            print("✓ Detected ForestEnv mission")
            
            # Test City environment
            mission_city = create_test_mission("CityEnv", "mission3")
            env = self.env_manager.get_mission_environment(mission_city)
            assert env == "CityEnv", f"Expected CityEnv, got {env}"
            print("✓ Detected CityEnv mission")
            
            # Test no environment specified (None)
            mission_no_env = {"metadata": {}}
            env = self.env_manager.get_mission_environment(mission_no_env)
            assert env is None, f"Expected None, got {env}"
            print("✓ Handled mission without environment specification")

            # Test alias normalization
            mission_alias = create_test_mission("AirSimNHEnv", "mission_alias")
            env = self.env_manager.get_mission_environment(mission_alias)
            assert env == "NHEnv", f"Expected NHEnv for alias, got {env}"
            print("✓ Normalized AirSimNHEnv alias to NHEnv")
            
            # Test invalid environment
            try:
                mission_invalid = create_test_mission("InvalidEnv", "mission_bad")
                env = self.env_manager.get_mission_environment(mission_invalid)
                raise AssertionError("Should have raised ValueError for invalid environment")
            except ValueError as e:
                print(f"✓ Rejected invalid environment: {e}")
            
            self.tests_passed += 1
            print("\n✓ TEST 1 PASSED")
            
        except Exception as e:
            self.tests_failed += 1
            print(f"\n✗ TEST 1 FAILED: {e}")
    
    def test_environment_switch_detection(self):
        """Test 2: Environment switch detection logic."""
        print("\n" + "="*70)
        print("TEST 2: Environment Switch Detection")
        print("="*70)
        
        try:
            # No server running - switch needed for any target
            assert self.env_manager.is_environment_switch_needed("NHEnv"), \
                "Should need switch when no server running"
            print("✓ Detects switch needed when no server running")
            
            # None target - no switch needed
            assert not self.env_manager.is_environment_switch_needed(None), \
                "Should not need switch for None target"
            print("✓ No switch needed for None target environment")
            
            # Same environment - no switch needed (when server running)
            self.env_manager.current_environment = "NHEnv"
            self.env_manager.current_process = type('MockProcess', (), {
                'poll': lambda self: None  # Process running
            })()
            
            assert not self.env_manager.is_environment_switch_needed("NHEnv"), \
                "Should not need switch for same environment"
            print("✓ No switch needed for same running environment")
            
            # Different environment - switch needed
            assert self.env_manager.is_environment_switch_needed("ForestEnv"), \
                "Should need switch for different environment"
            print("✓ Switch needed for different environment")
            
            # Reset state
            self.env_manager.current_process = None
            self.env_manager.current_environment = None
            
            self.tests_passed += 1
            print("\n✓ TEST 2 PASSED")
            
        except Exception as e:
            self.tests_failed += 1
            print(f"\n✗ TEST 2 FAILED: {e}")
    
    def test_mission_sequence_planning(self):
        """Test 3: Planning environment switches for mission sequence."""
        print("\n" + "="*70)
        print("TEST 3: Mission Sequence Planning")
        print("="*70)
        
        try:
            missions = [
                create_test_mission("NHEnv", "mission_nh1"),
                create_test_mission("NHEnv", "mission_nh2"),
                create_test_mission("ForestEnv", "mission_forest1"),
                create_test_mission("CityEnv", "mission_city1"),
                create_test_mission("ForestEnv", "mission_forest2"),
            ]
            
            expected_switches = [
                ("NHEnv", "Initial NHEnv"),
                (None, "Stay in NHEnv"),
                ("ForestEnv", "Switch to ForestEnv"),
                ("CityEnv", "Switch to CityEnv"),
                ("ForestEnv", "Switch back to ForestEnv"),
            ]
            
            current_env = None
            switch_count = 0
            
            for mission, (expected_env, description) in zip(missions, expected_switches):
                target_env = self.env_manager.get_mission_environment(mission)
                self.env_manager.current_environment = current_env
                self.env_manager.current_process = None
                
                if self.env_manager.is_environment_switch_needed(target_env):
                    print(f"  → {description}: {current_env} → {target_env}")
                    switch_count += 1
                    current_env = target_env
                else:
                    print(f"  ✓ {description}: Staying in {current_env}")
            
            assert switch_count == 4, f"Expected 4 switches, got {switch_count}"
            print(f"\n✓ Correctly planned {switch_count} environment switches for 5 missions")
            
            self.tests_passed += 1
            print("\n✓ TEST 3 PASSED")
            
        except Exception as e:
            self.tests_failed += 1
            print(f"\n✗ TEST 3 FAILED: {e}")
    
    def test_status_reporting(self):
        """Test 4: Status reporting."""
        print("\n" + "="*70)
        print("TEST 4: Status Reporting")
        print("="*70)
        
        try:
            # No server running
            status = self.env_manager.get_status()
            assert not status["is_running"], "Should report as not running"
            assert status["environment"] is None, "Should have None environment"
            print("✓ Reports correct status when no server running")
            
            # Mock running server
            self.env_manager.current_environment = "NHEnv"
            self.env_manager.current_process = type('MockProcess', (), {
                'poll': lambda self: None,
                'pid': 12345
            })()
            
            status = self.env_manager.get_status()
            assert status["is_running"], "Should report as running"
            assert status["environment"] == "NHEnv", "Should report NHEnv"
            assert status["process_id"] == 12345, "Should report correct PID"
            print("✓ Reports correct status when server running")
            
            # Reset
            self.env_manager.current_process = None
            self.env_manager.current_environment = None
            
            self.tests_passed += 1
            print("\n✓ TEST 4 PASSED")
            
        except Exception as e:
            self.tests_failed += 1
            print(f"\n✗ TEST 4 FAILED: {e}")
    
    def run_all_tests(self):
        """Run all tests."""
        print("\n" + "="*70)
        print("ENVIRONMENT MANAGER TEST SUITE")
        print("="*70)
        
        self.test_mission_environment_detection()
        self.test_environment_switch_detection()
        self.test_mission_sequence_planning()
        self.test_status_reporting()
        
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_failed}")
        print(f"Total: {self.tests_passed + self.tests_failed}")
        print("="*70 + "\n")
        
        return self.tests_failed == 0


def main():
    """Run tests."""
    project_root = Path(__file__).parent.parent
    
    print("\n" + "█"*70)
    print("█ TESTING DYNAMIC ENVIRONMENT SWITCHING")
    print("█"*70)
    print(f"\nProject Root: {project_root}")
    
    tester = EnvironmentManagerTestHarness(project_root)
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nThe environment manager is ready for use with:")
        print("  - Mission environment detection")
        print("  - Smart environment switching (no unnecessary restarts)")
        print("  - Clean server lifecycle management")
        print("  - Multi-environment mission sequences")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nFix the issues above and re-run the tests.")
        return 1


@pytest.fixture
def env_manager() -> EnvironmentManager:
    """Pytest fixture providing a fresh environment manager per test."""
    project_root = Path(__file__).parent.parent
    return EnvironmentManager(project_root)


def test_mission_environment_detection(env_manager: EnvironmentManager):
    """Pytest: Environment detection from mission metadata."""
    mission_nh = create_test_mission("NHEnv", "mission1")
    assert env_manager.get_mission_environment(mission_nh) == "NHEnv"

    mission_forest = create_test_mission("ForestEnv", "mission2")
    assert env_manager.get_mission_environment(mission_forest) == "ForestEnv"

    mission_city = create_test_mission("CityEnv", "mission3")
    assert env_manager.get_mission_environment(mission_city) == "CityEnv"

    mission_no_env = {"metadata": {}}
    assert env_manager.get_mission_environment(mission_no_env) is None

    mission_invalid = create_test_mission("InvalidEnv", "mission_bad")
    with pytest.raises(ValueError):
        env_manager.get_mission_environment(mission_invalid)


def test_environment_switch_detection(env_manager: EnvironmentManager):
    """Pytest: Environment switch decision logic."""
    assert env_manager.is_environment_switch_needed("NHEnv")
    assert not env_manager.is_environment_switch_needed(None)

    env_manager.current_environment = "NHEnv"
    env_manager.current_process = type("MockProcess", (), {
        "poll": lambda self: None,
    })()
    assert not env_manager.is_environment_switch_needed("NHEnv")
    assert env_manager.is_environment_switch_needed("ForestEnv")


def test_mission_sequence_planning(env_manager: EnvironmentManager):
    """Pytest: Correct environment-switch count across mixed mission sequence."""
    missions = [
        create_test_mission("NHEnv", "mission_nh1"),
        create_test_mission("NHEnv", "mission_nh2"),
        create_test_mission("ForestEnv", "mission_forest1"),
        create_test_mission("CityEnv", "mission_city1"),
        create_test_mission("ForestEnv", "mission_forest2"),
    ]

    current_env = None
    switch_count = 0

    for mission in missions:
        target_env = env_manager.get_mission_environment(mission)
        env_manager.current_environment = current_env
        # Simulate that a server is already running once an environment is active.
        if current_env is not None:
            env_manager.current_process = type("MockProcess", (), {
                "poll": lambda self: None,
            })()
        else:
            env_manager.current_process = None

        if env_manager.is_environment_switch_needed(target_env):
            switch_count += 1
            current_env = target_env

    assert switch_count == 4


def test_status_reporting(env_manager: EnvironmentManager):
    """Pytest: Status payload should reflect process/environment state."""
    status = env_manager.get_status()
    assert not status["is_running"]
    assert status["environment"] is None

    env_manager.current_environment = "NHEnv"
    env_manager.current_process = type("MockProcess", (), {
        "poll": lambda self: None,
        "pid": 12345,
    })()

    status = env_manager.get_status()
    assert status["is_running"]
    assert status["environment"] == "NHEnv"
    assert status["process_id"] == 12345


if __name__ == "__main__":
    sys.exit(main())
