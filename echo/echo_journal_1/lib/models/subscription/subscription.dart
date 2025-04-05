import 'plan.dart';

class Subscription {
  final int id;
  final int userId;
  final int planId;
  final Plan? planDetails;
  final String status;
  final DateTime startDate;
  final DateTime endDate;
  final bool isAutoRenewal;
  final DateTime createdAt;
  final DateTime updatedAt;
  final int daysRemaining;

  Subscription({
    required this.id,
    required this.userId,
    required this.planId,
    this.planDetails,
    required this.status,
    required this.startDate,
    required this.endDate,
    required this.isAutoRenewal,
    required this.createdAt,
    required this.updatedAt,
    required this.daysRemaining,
  });

  factory Subscription.fromJson(Map<String, dynamic> json) {
    if (json.isEmpty) {
      return Subscription(
        id: 0,
        userId: 0,
        planId: 0,
        planDetails: null,
        status: '',
        startDate: DateTime.now(),
        endDate: DateTime.now(),
        isAutoRenewal: false,
        createdAt: DateTime.now(),
        updatedAt: DateTime.now(),
        daysRemaining: 0,
      );
    }

    return Subscription(
      id: json['id'] ?? 0,
      userId: json['user'] ?? 0,
      planId: json['plan'] ?? 0,
      planDetails: json['plan_details'] != null
          ? Plan.fromJson(json['plan_details'])
          : null,
      status: json['status'] ?? '',
      startDate: json['start_date'] != null
          ? DateTime.parse(json['start_date'])
          : DateTime.now(),
      endDate: json['end_date'] != null
          ? DateTime.parse(json['end_date'])
          : DateTime.now(),
      isAutoRenewal: json['is_auto_renewal'] ?? false,
      createdAt: json['created_at'] != null
          ? DateTime.parse(json['created_at'])
          : DateTime.now(),
      updatedAt: json['updated_at'] != null
          ? DateTime.parse(json['updated_at'])
          : DateTime.now(),
      daysRemaining: json['days_remaining'] ?? 0,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'user': userId,
      'plan': planId,
      'plan_details': planDetails?.toJson(),
      'status': status,
      'start_date': startDate.toIso8601String(),
      'end_date': endDate.toIso8601String(),
      'is_auto_renewal': isAutoRenewal,
      'created_at': createdAt.toIso8601String(),
      'updated_at': updatedAt.toIso8601String(),
      'days_remaining': daysRemaining,
    };
  }

  bool get isActive => status == 'ACTIVE';
  bool get isPremium => isActive && planDetails?.planType == 'PREMIUM';
  bool get isFree => isActive && planDetails?.planType == 'FREE';
}
