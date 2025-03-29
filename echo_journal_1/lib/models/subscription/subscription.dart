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
    return Subscription(
      id: json['id'],
      userId: json['user'],
      planId: json['plan'],
      planDetails:
          json['plan_details'] != null
              ? Plan.fromJson(json['plan_details'])
              : null,
      status: json['status'],
      startDate: DateTime.parse(json['start_date']),
      endDate: DateTime.parse(json['end_date']),
      isAutoRenewal: json['is_auto_renewal'],
      createdAt: DateTime.parse(json['created_at']),
      updatedAt: DateTime.parse(json['updated_at']),
      daysRemaining: json['days_remaining'],
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
}
