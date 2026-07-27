// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {AccessControl} from "@openzeppelin/contracts/access/AccessControl.sol";
import {Pausable} from "@openzeppelin/contracts/utils/Pausable.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {Address} from "@openzeppelin/contracts/utils/Address.sol";

/**
 * @title MedicalEscrow
 * @notice Holds native MATIC for medical campaigns until an authorized oracle
 * approves a bill and an authorized release operator sends the payout.
 * @dev The off-chain OCR service must produce an auditable record. Only the
 * ORACLE_ROLE may submit that record's on-chain outcome; it cannot release
 * funds itself.
 */
contract MedicalEscrow is AccessControl, Pausable, ReentrancyGuard {
    using Address for address payable;

    bytes32 public constant ORACLE_ROLE = keccak256("ORACLE_ROLE");
    bytes32 public constant RELEASE_ROLE = keccak256("RELEASE_ROLE");
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");

    address public oracle;
    bytes32 public jobId;
    uint256 public fee;
    uint256 public campaignCount;

    enum VerificationStatus {
        Pending,
        Approved,
        Rejected
    }

    struct Campaign {
        address payable creator;
        address payable hospitalWallet;
        uint256 targetAmount;
        uint256 amountRaised;
        uint256 billTotalExtracted;
        uint256 fraudRiskScore;
        VerificationStatus verificationStatus;
        bool isReleased;
    }

    mapping(uint256 => Campaign) public campaigns;
    mapping(uint256 => mapping(address => uint256)) public contributions;

    event CampaignCreated(
        uint256 indexed campaignId,
        address indexed creator,
        address indexed hospital,
        uint256 targetAmount
    );
    event DonationReceived(uint256 indexed campaignId, address indexed donor, uint256 amount);
    event BillVerificationRequested(uint256 indexed campaignId, bytes32 indexed requestId);
    event BillVerificationFulfilled(
        uint256 indexed campaignId,
        bool isVerified,
        uint256 billTotal,
        uint256 fraudRiskScore
    );
    event FundsReleased(uint256 indexed campaignId, address indexed hospital, uint256 amount);
    event RefundIssued(uint256 indexed campaignId, address indexed donor, uint256 amount);

    error CampaignDoesNotExist(uint256 campaignId);
    error InvalidHospitalAddress();
    error InvalidTargetAmount();
    error InvalidDonationAmount();
    error CampaignAlreadyReleased(uint256 campaignId);
    error CampaignNotApproved(uint256 campaignId);
    error CampaignNotRejected(uint256 campaignId);
    error CampaignDoesNotMeetReleaseRequirements(uint256 campaignId);
    error NoFundsAvailable(uint256 campaignId);
    error NoRefundAvailable(uint256 campaignId, address donor);

    constructor(address initialOracle, bytes32 _jobId, uint256 _fee) {
        if (initialOracle == address(0)) revert InvalidHospitalAddress();

        oracle = initialOracle;
        jobId = _jobId;
        fee = _fee;

        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ORACLE_ROLE, initialOracle);
        _grantRole(RELEASE_ROLE, msg.sender);
        _grantRole(PAUSER_ROLE, msg.sender);
    }

    function createCampaign(address payable hospitalWallet, uint256 targetAmount)
        external
        whenNotPaused
        returns (uint256 campaignId)
    {
        if (hospitalWallet == address(0)) revert InvalidHospitalAddress();
        if (targetAmount == 0) revert InvalidTargetAmount();

        campaignId = ++campaignCount;
        campaigns[campaignId] = Campaign({
            creator: payable(msg.sender),
            hospitalWallet: hospitalWallet,
            targetAmount: targetAmount,
            amountRaised: 0,
            billTotalExtracted: 0,
            fraudRiskScore: type(uint256).max,
            verificationStatus: VerificationStatus.Pending,
            isReleased: false
        });

        emit CampaignCreated(campaignId, msg.sender, hospitalWallet, targetAmount);
    }

    function donate(uint256 campaignId) external payable whenNotPaused nonReentrant {
        Campaign storage campaign = _campaign(campaignId);
        if (campaign.isReleased) revert CampaignAlreadyReleased(campaignId);
        if (msg.value == 0) revert InvalidDonationAmount();

        campaign.amountRaised += msg.value;
        contributions[campaignId][msg.sender] += msg.value;

        emit DonationReceived(campaignId, msg.sender, msg.value);
    }

    /**
     * @notice Creates an auditable request marker for an off-chain bill review.
     * @dev The request is intentionally role-restricted so arbitrary users
     * cannot create misleading oracle audit events.
     */
    function requestBillVerification(uint256 campaignId, string calldata billReference)
        external
        onlyRole(ORACLE_ROLE)
        whenNotPaused
        returns (bytes32 requestId)
    {
        Campaign storage campaign = _campaign(campaignId);
        if (campaign.isReleased) revert CampaignAlreadyReleased(campaignId);
        requestId = keccak256(
            abi.encodePacked(block.chainid, address(this), campaignId, billReference, block.timestamp)
        );
        emit BillVerificationRequested(campaignId, requestId);
    }

    /**
     * @notice Records the audited OCR/fraud decision. This never transfers funds.
     */
    function fulfillBillVerification(
        uint256 campaignId,
        bool isVerified,
        uint256 billTotalExtracted,
        uint256 fraudRiskScore
    ) external onlyRole(ORACLE_ROLE) whenNotPaused {
        Campaign storage campaign = _campaign(campaignId);
        if (campaign.isReleased) revert CampaignAlreadyReleased(campaignId);

        campaign.billTotalExtracted = billTotalExtracted;
        campaign.fraudRiskScore = fraudRiskScore;
        campaign.verificationStatus = isVerified
            ? VerificationStatus.Approved
            : VerificationStatus.Rejected;

        emit BillVerificationFulfilled(campaignId, isVerified, billTotalExtracted, fraudRiskScore);
    }

    /**
     * @notice Sends locked campaign funds after independent oracle approval.
     */
    function releaseFunds(uint256 campaignId)
        external
        onlyRole(RELEASE_ROLE)
        whenNotPaused
        nonReentrant
    {
        Campaign storage campaign = _campaign(campaignId);
        if (campaign.isReleased) revert CampaignAlreadyReleased(campaignId);
        if (campaign.verificationStatus != VerificationStatus.Approved) {
            revert CampaignNotApproved(campaignId);
        }
        if (
            campaign.billTotalExtracted < campaign.targetAmount ||
            campaign.fraudRiskScore >= 10
        ) {
            revert CampaignDoesNotMeetReleaseRequirements(campaignId);
        }
        if (campaign.amountRaised == 0) revert NoFundsAvailable(campaignId);

        uint256 payout = campaign.amountRaised;
        campaign.amountRaised = 0;
        campaign.isReleased = true;
        campaign.hospitalWallet.sendValue(payout);

        emit FundsReleased(campaignId, campaign.hospitalWallet, payout);
    }

    /**
     * @notice Returns a donor's contribution only after an explicit rejection.
     */
    function claimRefund(uint256 campaignId) external nonReentrant {
        Campaign storage campaign = _campaign(campaignId);
        if (campaign.isReleased) revert CampaignAlreadyReleased(campaignId);
        if (campaign.verificationStatus != VerificationStatus.Rejected) {
            revert CampaignNotRejected(campaignId);
        }

        uint256 donation = contributions[campaignId][msg.sender];
        if (donation == 0) revert NoRefundAvailable(campaignId, msg.sender);

        contributions[campaignId][msg.sender] = 0;
        campaign.amountRaised -= donation;
        payable(msg.sender).sendValue(donation);

        emit RefundIssued(campaignId, msg.sender, donation);
    }

    function pause() external onlyRole(PAUSER_ROLE) {
        _pause();
    }

    function unpause() external onlyRole(PAUSER_ROLE) {
        _unpause();
    }

    function _campaign(uint256 campaignId) private view returns (Campaign storage campaign) {
        campaign = campaigns[campaignId];
        if (campaign.creator == address(0)) revert CampaignDoesNotExist(campaignId);
    }
}
