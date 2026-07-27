// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {AccessControl} from "@openzeppelin/contracts/access/AccessControl.sol";

/**
 * @title TimelockController
 * @notice Governance timelock for delaying sensitive MedicalEscrow operations
 *         such as oracle address changes, pausing, or role revocations.
 * @dev All PROPOSER_ROLE members can schedule; EXECUTOR_ROLE can execute after
 *         minimum delay. The OWNER_ROLE can cancel and override.
 */
contract TimelockController is AccessControl {
    bytes32 public constant PROPOSER_ROLE = keccak256("PROPOSER_ROLE");
    bytes32 public constant EXECUTOR_ROLE = keccak256("EXECUTOR_ROLE");
    bytes32 public constant CANCELLER_ROLE = keccak256("CANCELLER_ROLE");

    error Timelock__InvalidDelay();
    error Timelock__InvalidTarget();
    error Timelock__InvalidOperation(bytes32 operationId);
    error Timelock__OperationAlreadyScheduled(bytes32 operationId);
    error Timelock__OperationNotReady(bytes32 operationId);
    error Timelock__OperationAlreadyExecuted(bytes32 operationId);
    error Timelock__CallFailed(bytes32 operationId);

    event CallScheduled(
        bytes32 indexed operationId,
        address indexed target,
        uint256 value,
        bytes data,
        bytes32 predecessor,
        uint256 delay
    );
    event CallExecuted(
        bytes32 indexed operationId,
        address indexed target,
        uint256 value,
        bytes data
    );
    event CallCancelled(bytes32 indexed operationId);

    uint256 private _minDelay;
    mapping(bytes32 => uint256) private _timestamps;

    /**
     * @param minDelay Minimum delay in seconds before an operation can be executed.
     * @param proposers List of addresses to grant PROPOSER_ROLE.
     * @param executors List of addresses to grant EXECUTOR_ROLE.
     * @param admin The address that will be granted DEFAULT_ADMIN_ROLE.
     */
    constructor(
        uint256 minDelay,
        address[] memory proposers,
        address[] memory executors,
        address admin
    ) {
        if (minDelay < 1 hours) revert Timelock__InvalidDelay();

        _minDelay = minDelay;

        _grantRole(DEFAULT_ADMIN_ROLE, admin);

        for (uint256 i = 0; i < proposers.length; ++i) {
            _grantRole(PROPOSER_ROLE, proposers[i]);
        }

        for (uint256 i = 0; i < executors.length; ++i) {
            _grantRole(EXECUTOR_ROLE, executors[i]);
        }

        // Admin is also cancellor
        _grantRole(CANCELLER_ROLE, admin);
    }

    receive() external payable {}

    // ─── Public Getters ───────────────────────────────────────────────

    function getMinDelay() external view returns (uint256) {
        return _minDelay;
    }

    function getTimestamp(bytes32 operationId) external view returns (uint256) {
        return _timestamps[operationId];
    }

    function isOperationPending(bytes32 operationId) public view returns (bool) {
        return _timestamps[operationId] > 0;
    }

    function isOperationReady(bytes32 operationId) public view returns (bool) {
        uint256 timestamp = _timestamps[operationId];
        return timestamp > 0 && block.timestamp >= timestamp;
    }

    function isOperationDone(bytes32 operationId) public view returns (bool) {
        return _timestamps[operationId] == 1;
    }

    // ─── Scheduling ──────────────────────────────────────────────────

    function hashOperation(
        address target,
        uint256 value,
        bytes calldata data,
        bytes32 predecessor,
        bytes32 salt
    ) public pure returns (bytes32) {
        return keccak256(abi.encode(target, value, data, predecessor, salt));
    }

    function schedule(
        address target,
        uint256 value,
        bytes calldata data,
        bytes32 predecessor,
        bytes32 salt,
        uint256 delay
    ) external onlyRole(PROPOSER_ROLE) {
        if (target == address(0)) revert Timelock__InvalidTarget();
        if (delay < _minDelay) revert Timelock__InvalidDelay();

        bytes32 operationId = hashOperation(target, value, data, predecessor, salt);

        if (_timestamps[operationId] != 0) {
            revert Timelock__OperationAlreadyScheduled(operationId);
        }

        _timestamps[operationId] = block.timestamp + delay;

        emit CallScheduled(operationId, target, value, data, predecessor, delay);
    }

    // ─── Execution ───────────────────────────────────────────────────

    function execute(
        address target,
        uint256 value,
        bytes calldata data,
        bytes32 predecessor,
        bytes32 salt
    ) external payable onlyRole(EXECUTOR_ROLE) {
        bytes32 operationId = hashOperation(target, value, data, predecessor, salt);

        if (_timestamps[operationId] == 0) {
            revert Timelock__InvalidOperation(operationId);
        }
        if (_timestamps[operationId] == 1) {
            revert Timelock__OperationAlreadyExecuted(operationId);
        }
        if (block.timestamp < _timestamps[operationId]) {
            revert Timelock__OperationNotReady(operationId);
        }

        if (predecessor != bytes32(0)) {
            if (!isOperationDone(predecessor)) {
                revert Timelock__OperationNotReady(predecessor);
            }
        }

        // Mark as executed
        _timestamps[operationId] = 1;

        (bool success, ) = target.call{value: value}(data);
        if (!success) {
            revert Timelock__CallFailed(operationId);
        }

        emit CallExecuted(operationId, target, value, data);
    }

    // ─── Cancellation ────────────────────────────────────────────────

    function cancel(bytes32 operationId) external onlyRole(CANCELLER_ROLE) {
        if (_timestamps[operationId] == 0) {
            revert Timelock__InvalidOperation(operationId);
        }
        if (_timestamps[operationId] == 1) {
            revert Timelock__OperationAlreadyExecuted(operationId);
        }

        delete _timestamps[operationId];

        emit CallCancelled(operationId);
    }

    // ─── Admin ───────────────────────────────────────────────────────

    function updateDelay(uint256 newDelay) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (newDelay < 1 hours) revert Timelock__InvalidDelay();
        _minDelay = newDelay;
    }
}

