from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import UserData
from django.views.decorators.http import require_http_methods
from rest_framework.decorators import api_view 
import json
from .models import SegmentHistory

#로그인
@api_view(['POST'])
@csrf_exempt
def save_user(request):
    user_id = request.data.get('user_id')
    password = request.data.get('password')

    try:
        person = UserData.objects.get(user_id=user_id)

        if person.password == password:
            return JsonResponse({
                "message": "로그인 성공",
            }, status=200)

        else:
            return JsonResponse({"error": "비밀번호가 일치하지 않습니다."})

    except UserData.DoesNotExist:
        return JsonResponse({"error": "존재하지 않는 사용자입니다."})

#세그먼트 저장
@csrf_exempt
@require_http_methods(["POST"])
def save_segment(request):
    try:
        body = json.loads(request.body or "{}")
    except Exception:
        return JsonResponse({"error": "JSON 파싱 실패"}, status=400)

    # ✅ 로그인한 사용자 id (프론트에서 같이 보내줌)
    user_id      = (body.get("user_id") or "").strip()
    segment_name = (body.get("segment_name") or "").strip()
    user_input   = (body.get("user_input") or "").strip()
    stage3       = body.get("stage3")
    insight      = body.get("insight")

    if not user_id:
        return JsonResponse({"error": "user_id가 필요합니다."}, status=400)

    if not segment_name:
        return JsonResponse({"error": "segment_name은 필수입니다."}, status=400)

    if not isinstance(stage3, dict):
        return JsonResponse({"error": "stage3는 dict여야 합니다."}, status=400)

    if not isinstance(insight, dict):
        return JsonResponse({"error": "insight는 dict여야 합니다."}, status=400)

    seg = SegmentHistory.objects.create(
        user_id=user_id,                
        segment_name=segment_name,
        user_input=user_input,
        main=stage3.get("main"),
        sub=stage3.get("sub"),
        stage3=stage3,
        insight=insight,
    )

    return JsonResponse(
        {"message": "세그먼트 저장 완료", "id": seg.id},
        status=201,
        json_dumps_params={"ensure_ascii": False},
    )

#카드 리스트 보여줄때
@require_http_methods(["GET"])
def list_segments(request):
    user_id = request.GET.get("user_id")

    if not user_id:
        return JsonResponse({"error": "user_id가 필요합니다."}, status=400)

    qs = SegmentHistory.objects.filter(user_id=user_id).order_by("-created_at")

    items = []
    for seg in qs:
        stage3 = seg.stage3 or {}
        items.append({
            "id": seg.id,
            "segment_name": seg.segment_name,
            "count": stage3.get("count"),
            "main": stage3.get("main") or seg.main,
            "sub": stage3.get("sub") or seg.sub,
        })

    return JsonResponse({"segments": items},
                        json_dumps_params={"ensure_ascii": False})

#세그먼트 인사이트 보여줄때
@require_http_methods(["GET"])
def retrieve_segment(request, segment_id):
    user_id = request.GET.get("user_id")
    if not user_id:
        return JsonResponse({"error": "user_id가 필요합니다."}, status=400)

    try:
        seg = SegmentHistory.objects.get(id=segment_id, user_id=user_id)
    except SegmentHistory.DoesNotExist:
        return JsonResponse({"error": "Segment not found"}, status=404)

    stage3 = seg.stage3 or {}

    return JsonResponse({
        "id": seg.id,
        "segment_name": seg.segment_name,
        "user_input": seg.user_input,
        "main": stage3.get("main") or seg.main,
        "sub": stage3.get("sub") or seg.sub,
        "stage3": stage3,
        "insight": seg.insight,
    }, json_dumps_params={"ensure_ascii": False})
