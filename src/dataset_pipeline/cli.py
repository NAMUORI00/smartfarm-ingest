#!/usr/bin/env python3
"""
Dataset Pipeline CLI - 통합 명령줄 인터페이스

기존 corpus_cli.py 및 scripts/*.py의 기능을 통합한 CLI.
ConfigManager를 통해 설정을 관리하고, 기존 모듈을 재사용합니다.

Usage:
    # 도움말
    python -m dataset_pipeline.cli --help
    
    # 웹 크롤링
    python -m dataset_pipeline.cli crawl-wasabi --output output/wasabi_web_en.jsonl
    
    # 번역
    python -m dataset_pipeline.cli translate --input output/wasabi_web_en.jsonl --output output/wasabi_en_ko.jsonl
    
    # QA 생성
    python -m dataset_pipeline.cli generate-qa --input output/wasabi_en_ko.jsonl --output output/wasabi_qa.jsonl
    
    # MQM 평가
    python -m dataset_pipeline.cli mqm-score --input output/wasabi_en_ko.jsonl --output output/wasabi_scored.jsonl
    
    # 전체 파이프라인
    python -m dataset_pipeline.cli run-pipeline --corpus wasabi --output-dir output/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click

from .config import ConfigManager, get_config
from .constants import QAFields, CorpusFields


# 프로젝트 루트 기준 기본 경로
DEFAULT_OUTPUT_DIR = "output"


def _ensure_config(config_path: Optional[str] = None) -> ConfigManager:
    """ConfigManager 인스턴스 생성 또는 반환."""
    if config_path:
        return ConfigManager(config_path=config_path)
    return get_config()


@click.group()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    default=None,
    help="설정 파일 경로 (기본: config/settings.yaml)",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="상세 출력 모드",
)
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], verbose: bool):
    """Dataset Pipeline CLI - 스마트팜 데이터셋 구축 도구.
    
    환경변수 또는 config/secrets.yaml을 통해 API 키를 설정하세요.
    
    \b
    예시:
        export API_KEY="your-api-key"
        export OPENAI_BASE_URL="https://api.openai.com/v1"
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = _ensure_config(config)
    ctx.obj["verbose"] = verbose


# =============================================================================
# crawl-wasabi 명령
# =============================================================================

@cli.command("crawl-wasabi")
@click.option("--output", "-o", required=True, help="출력 JSONL 파일 경로")
@click.option("--limit", type=int, default=None, help="최대 크롤링 문서 수")
@click.option("--delay", type=float, default=1.0, help="요청 간 대기 시간 (초)")
@click.option("--granularity", type=click.Choice(["chunk", "doc"]), default="chunk", help="출력 단위")
@click.option("--chunk-size", type=int, default=None, help="청크 크기 (기본: 설정 파일)")
@click.option("--chunk-overlap", type=int, default=None, help="청크 오버랩 (기본: 설정 파일)")
@click.option("--resume", is_flag=True, help="기존 출력 파일에서 이어서 진행")
@click.pass_context
def crawl_wasabi(
    ctx: click.Context,
    output: str,
    limit: Optional[int],
    delay: float,
    granularity: str,
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    resume: bool,
):
    """와사비 관련 웹 문서 크롤링.
    
    Wikipedia, Britannica 등 신뢰할 수 있는 소스에서 와사비 재배 정보를 수집합니다.
    """
    config: ConfigManager = ctx.obj["config"]
    
    # 기존 corpus_cli의 cmd_crawl_wasabi 호출
    from .corpus_cli import cmd_crawl_wasabi
    import argparse
    
    rag_cfg = config.get("rag", {})
    
    args = argparse.Namespace(
        config=str(config.config_path),
        output=output,
        limit=limit,
        delay=delay,
        granularity=granularity,
        chunk_size=chunk_size or rag_cfg.get("chunk_size", 512),
        chunk_overlap=chunk_overlap or rag_cfg.get("chunk_overlap", 50),
        resume=resume,
        flush_every=50,
    )
    
    return cmd_crawl_wasabi(args)


# =============================================================================
# translate 명령
# =============================================================================

@cli.command("translate")
@click.option("--input", "-i", "input_file", required=True, help="입력 JSONL 파일 경로")
@click.option("--output", "-o", required=True, help="출력 JSONL 파일 경로")
@click.option("--src-lang", default="en", help="원본 언어 (기본: en)")
@click.option("--tgt-lang", default="ko", help="대상 언어 (기본: ko)")
@click.option("--glossary", default=None, help="용어집 YAML/JSON 파일 경로")
@click.option("--temperature", type=float, default=0.0, help="LLM 온도 (기본: 0.0)")
@click.option("--max-tokens", type=int, default=2048, help="최대 토큰 수")
@click.option("--sleep", type=float, default=0.0, help="요청 간 대기 시간 (초)")
@click.option("--resume", is_flag=True, help="기존 출력 파일에서 이어서 진행")
@click.pass_context
def translate(
    ctx: click.Context,
    input_file: str,
    output: str,
    src_lang: str,
    tgt_lang: str,
    glossary: Optional[str],
    temperature: float,
    max_tokens: int,
    sleep: float,
    resume: bool,
):
    """EN→KO 번역 (LLM 기반).
    
    Generator 모델을 사용하여 영문 코퍼스를 한국어로 번역합니다.
    """
    config: ConfigManager = ctx.obj["config"]
    
    from .corpus_cli import cmd_translate
    import argparse
    
    args = argparse.Namespace(
        config=str(config.config_path),
        input=input_file,
        output=output,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        glossary=glossary,
        temperature=temperature,
        max_tokens=max_tokens,
        sleep=sleep,
        resume=resume,
        flush_every=20,
    )
    
    return cmd_translate(args)


# =============================================================================
# mqm-score 명령
# =============================================================================

@cli.command("mqm-score")
@click.option("--input", "-i", "input_file", required=True, help="입력 JSONL 파일 경로 (번역된 병렬 코퍼스)")
@click.option("--output", "-o", required=True, help="출력 JSONL 파일 경로")
@click.option("--src-lang", default="en", help="원본 언어 (기본: en)")
@click.option("--tgt-lang", default="ko", help="대상 언어 (기본: ko)")
@click.option("--temperature", type=float, default=0.0, help="Judge 온도 (기본: 0.0)")
@click.option("--sleep", type=float, default=0.0, help="요청 간 대기 시간 (초)")
@click.option("--resume", is_flag=True, help="기존 출력 파일에서 이어서 진행")
@click.pass_context
def mqm_score(
    ctx: click.Context,
    input_file: str,
    output: str,
    src_lang: str,
    tgt_lang: str,
    temperature: float,
    sleep: float,
    resume: bool,
):
    """MQM 스타일 번역 품질 평가 (LLM-as-a-Judge).
    
    설정된 Judge 모델(들)을 사용하여 번역 품질을 평가합니다.
    """
    config: ConfigManager = ctx.obj["config"]
    
    from .corpus_cli import cmd_mqm_score
    import argparse
    
    args = argparse.Namespace(
        config=str(config.config_path),
        input=input_file,
        output=output,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        temperature=temperature,
        sleep=sleep,
        resume=resume,
        flush_every=20,
    )
    
    return cmd_mqm_score(args)


# =============================================================================
# generate-qa 명령
# =============================================================================

@cli.command("generate-qa")
@click.option("--input", "-i", "input_file", required=True, help="입력 JSONL 파일 경로 (번역된 코퍼스)")
@click.option("--output", "-o", required=True, help="출력 JSONL 파일 경로")
@click.option("--num-questions", "-n", type=int, default=220, help="생성할 QA 쌍 수 (기본: 220)")
@click.option("--model", default=None, help="LLM 모델명 (기본: 설정 파일의 generator)")
@click.option("--lexical-threshold", type=float, default=None, help="질문-문맥 lexical overlap 임계값 (예: 0.2). 설정 시 초과 질문 제거")
@click.option("--paraphrase", is_flag=True, help="질문 패러프레이즈 적용")
@click.option("--paraphrase-temp", type=float, default=0.3, help="패러프레이즈 온도 (기본: 0.3)")
@click.option("--multi-source", is_flag=True, help="동일 텍스트 문서 ID를 source_ids에 함께 포함")
@click.option("--delay", type=float, default=0.5, help="요청 간 대기 시간 (초)")
@click.option("--resume", is_flag=True, help="기존 출력 파일에서 이어서 진행")
@click.pass_context
def generate_qa(
    ctx: click.Context,
    input_file: str,
    output: str,
    num_questions: int,
    model: Optional[str],
    lexical_threshold: Optional[float],
    paraphrase: bool,
    paraphrase_temp: float,
    multi_source: bool,
    delay: float,
    resume: bool,
):
    """QA 데이터셋 생성.
    
    번역된 코퍼스에서 Self-Instruct 방식으로 질문-답변 쌍을 생성합니다.
    
    \b
    생성되는 질문 유형:
    - basic: 단순 사실 질문 (온도, 수치, 이름)
    - intermediate: 관계 추론 질문 (원인-결과, 비교)
    - advanced: 다단계 복합 질문 (문제해결, 최적화)
    """
    config: ConfigManager = ctx.obj["config"]
    verbose = ctx.obj["verbose"]
    
    # 직접 구현 (scripts/generate_wasabi_qa.py 로직 통합)
    import json
    import random
    import time
    from pathlib import Path
    from openai import OpenAI
    
    llm_config = config.get_llm_config("generator")
    actual_model = model or llm_config.get("model", "gemini-2.5-flash")
    
    client = OpenAI(
        base_url=llm_config.get("base_url"),
        api_key=llm_config.get("api_key"),
    )
    
    input_path = Path(input_file)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 코퍼스 로드
    corpus = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    
    click.echo(f"코퍼스 로드 완료: {len(corpus)}개 문서")
    
    # 기존 QA 로드 (resume 모드)
    existing_questions = []
    existing_ids = set()
    if resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                existing_questions.append(row[QAFields.QUESTION])
                existing_ids.add(row[QAFields.ID])
        click.echo(f"기존 QA에서 이어서 진행: {len(existing_ids)}개")
    
    # 질문 카테고리 및 난이도
    categories = ["환경조건", "양액관리", "병해충", "재배기술", "수확품질", "설비장비"]
    complexities = {
        "basic": "단순 사실 질문",
        "intermediate": "관계 추론 질문",
        "advanced": "다단계 복합 질문",
    }
    
    # 질문 생성 프롬프트
    extra_rules = ""
    if lexical_threshold is not None or paraphrase:
        extra_rules = "\n- 원문 표현을 그대로 복사하지 말고 의미를 유지한 재구성 문장으로 질문 작성"
    q_prompt_template = """당신은 와사비 재배 전문가입니다. 주어진 문서를 바탕으로 교육적 가치가 높은 질문을 생성하세요.

[문서]
{context}

[기존 질문들 (중복 피하기)]
{existing}

[요구사항]
- 문서 내용에 직접 근거하는 질문만 생성
- 질문 유형: {complexity}
- 카테고리: {category}
- 한국어로 3개 질문 생성
{extra_rules}

JSON 배열로 출력:
[{{"question": "질문", "answer_hint": "힌트"}}]"""
    
    # 답변 생성 프롬프트
    a_prompt_template = """당신은 와사비 재배 전문가입니다. 주어진 문서를 바탕으로 질문에 정확하게 답변하세요.

[질문]
{question}

[참고 문서]
{context}

[요구사항]
- 문서에 근거한 답변만 제공 (환각 금지)
- 수치, 단위, 조건을 정확히 인용
- 2-4문장으로 간결하게 답변

답변:"""
    
    def normalize_text(text: str) -> str:
        import re
        text = (text or "").lower()
        text = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def char_ngrams(text: str, n: int = 3) -> set:
        text = normalize_text(text).replace(" ", "")
        if len(text) < n:
            return set()
        return {text[i:i + n] for i in range(len(text) - n + 1)}

    def jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    # 동일 텍스트 문서 ID 매핑 (multi_source 옵션)
    text_to_ids = {}
    if multi_source:
        text_to_ids = {}
        for doc in corpus:
            text_ko = doc.get(CorpusFields.TEXT_KO) or doc.get(CorpusFields.TEXT, "")
            key = normalize_text(text_ko)
            if key:
                text_to_ids.setdefault(key, []).append(doc.get(CorpusFields.ID, "unknown"))

    qa_pairs = []
    qa_id = len(existing_ids)
    random.shuffle(corpus)
    
    click.echo(f"QA 생성 시작: 목표 {num_questions}개")
    
    with click.progressbar(length=num_questions - len(existing_ids), label="QA 생성") as bar:
        for doc in corpus:
            if len(qa_pairs) + len(existing_ids) >= num_questions:
                break

            text_ko = doc.get(CorpusFields.TEXT_KO) or doc.get(CorpusFields.TEXT, "")
            if len(text_ko) < 100:
                continue

            doc_id = doc.get(CorpusFields.ID, "unknown")
            category = random.choice(categories)
            complexity_key = random.choice(list(complexities.keys()))
            
            # 질문 생성
            q_prompt = q_prompt_template.format(
                context=text_ko[:2000],
                existing="\n".join(existing_questions[-10:]) if existing_questions else "(없음)",
                complexity=complexities[complexity_key],
                category=category,
                extra_rules=extra_rules,
            )
            
            try:
                resp = client.chat.completions.create(
                    model=actual_model,
                    messages=[{"role": "user", "content": q_prompt}],
                    temperature=0.7,
                    max_tokens=1000,
                )
                content = resp.choices[0].message.content
                
                import re
                json_match = re.search(r'\[[\s\S]*\]', content)
                questions = json.loads(json_match.group()) if json_match else []
            except Exception as e:
                if verbose:
                    click.echo(f"\n질문 생성 오류: {e}", err=True)
                continue
            
            context_ngrams = char_ngrams(text_ko[:2000]) if lexical_threshold is not None else set()

            for q_data in questions:
                if len(qa_pairs) + len(existing_ids) >= num_questions:
                    break
                
                question = q_data.get("question", "").strip()
                if not question or question in existing_questions:
                    continue

                original_question = question
                paraphrased = False
                if paraphrase:
                    paraphrase_prompt = (
                        "다음 질문을 의미를 유지하면서 다른 표현으로 바꿔주세요. "
                        "전문 용어와 핵심 키워드는 유지하고, 질문만 출력하세요.\n\n"
                        f"[질문]\n{question}\n"
                    )
                    try:
                        p_resp = client.chat.completions.create(
                            model=actual_model,
                            messages=[{"role": "user", "content": paraphrase_prompt}],
                            temperature=paraphrase_temp,
                            max_tokens=200,
                        )
                        p_text = p_resp.choices[0].message.content.strip()
                        if p_text:
                            question = p_text
                            paraphrased = True
                    except Exception as e:
                        if verbose:
                            click.echo(f"\n패러프레이즈 오류: {e}", err=True)

                if question in existing_questions:
                    continue

                lexical_overlap = None
                if lexical_threshold is not None and context_ngrams:
                    q_ngrams = char_ngrams(question)
                    if q_ngrams:
                        lexical_overlap = jaccard(q_ngrams, context_ngrams)
                        if lexical_overlap >= lexical_threshold:
                            continue
                
                # 답변 생성
                a_prompt = a_prompt_template.format(question=question, context=text_ko[:3000])
                
                try:
                    resp = client.chat.completions.create(
                        model=actual_model,
                        messages=[{"role": "user", "content": a_prompt}],
                        temperature=0.0,
                        max_tokens=500,
                    )
                    answer = resp.choices[0].message.content.strip()
                except Exception as e:
                    if verbose:
                        click.echo(f"\n답변 생성 오류: {e}", err=True)
                    continue
                
                if not answer or "문서에 해당 정보가 없습니다" in answer:
                    continue
                
                source_ids = [doc_id]
                if multi_source:
                    key = normalize_text(text_ko)
                    if key in text_to_ids:
                        source_ids = sorted(set(text_to_ids[key]))

                qa_pair = {
                    QAFields.ID: f"wasabi_qa_{qa_id:04d}",
                    QAFields.QUESTION: question,
                    QAFields.ANSWER: answer,
                    QAFields.CONTEXT: text_ko[:1000],
                    QAFields.CATEGORY: category,
                    QAFields.COMPLEXITY: complexity_key,
                    QAFields.SOURCE_IDS: source_ids,
                    "metadata": {
                        "answer_hint": q_data.get("answer_hint", ""),
                        "model": actual_model,
                        "paraphrased": paraphrased,
                        "original_question": original_question if paraphrased else "",
                        "lexical_overlap": lexical_overlap if lexical_overlap is not None else "",
                        "lexical_threshold": lexical_threshold if lexical_threshold is not None else "",
                        "source_id_equivalence_count": len(source_ids),
                    },
                }
                
                qa_pairs.append(qa_pair)
                existing_questions.append(question)
                qa_id += 1
                
                # 저장
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
                
                bar.update(1)
                time.sleep(delay)
    
    click.echo(f"\n생성 완료: 총 {len(qa_pairs) + len(existing_ids)}개 QA 쌍")
    click.echo(f"출력 파일: {output_path}")
    return 0


# =============================================================================
# export-cgiar 명령
# =============================================================================

@cli.command("export-cgiar")
@click.option("--output", "-o", required=True, help="출력 JSONL 파일 경로")
@click.option("--datasets", multiple=True, default=None, help="HuggingFace 데이터셋 repo_id (여러 개 가능)")
@click.option("--split", default="train", help="데이터셋 split (기본: train)")
@click.option("--limit-per-dataset", type=int, default=None, help="데이터셋당 최대 문서 수")
@click.option("--granularity", type=click.Choice(["chunk", "doc"]), default="chunk", help="출력 단위")
@click.option("--chunk-size", type=int, default=None, help="청크 크기")
@click.option("--chunk-overlap", type=int, default=None, help="청크 오버랩")
@click.option("--filter-keyword", multiple=True, default=[], help="필터 키워드 (여러 개 가능)")
@click.option("--no-streaming", is_flag=True, help="스트리밍 비활성화 (전체 다운로드)")
@click.option("--resume", is_flag=True, help="기존 출력 파일에서 이어서 진행")
@click.pass_context
def export_cgiar(
    ctx: click.Context,
    output: str,
    datasets: tuple,
    split: str,
    limit_per_dataset: Optional[int],
    granularity: str,
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    filter_keyword: tuple,
    no_streaming: bool,
    resume: bool,
):
    """CGIAR 공식 데이터셋에서 EN 코퍼스 추출.
    
    HuggingFace의 CGIAR 공식 데이터셋에서 농업 문서를 추출합니다.
    """
    config: ConfigManager = ctx.obj["config"]
    
    from .corpus_cli import cmd_export_cgiar
    import argparse
    
    args = argparse.Namespace(
        config=str(config.config_path),
        output=output,
        datasets=list(datasets) if datasets else None,
        split=split,
        limit_per_dataset=limit_per_dataset,
        granularity=granularity,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        filter_keyword=list(filter_keyword),
        no_streaming=no_streaming,
        resume=resume,
        flush_every=200,
    )
    
    return cmd_export_cgiar(args)


# =============================================================================
# postedit 명령
# =============================================================================

@cli.command("postedit")
@click.option("--input", "-i", "input_file", required=True, help="입력 JSONL 파일 경로")
@click.option("--output", "-o", required=True, help="출력 JSONL 파일 경로")
@click.option("--temperature", type=float, default=0.0, help="LLM 온도")
@click.option("--max-tokens", type=int, default=1800, help="최대 토큰 수")
@click.option("--sleep", type=float, default=0.0, help="요청 간 대기 시간 (초)")
@click.option("--resume", is_flag=True, help="기존 출력 파일에서 이어서 진행")
@click.pass_context
def postedit(
    ctx: click.Context,
    input_file: str,
    output: str,
    temperature: float,
    max_tokens: int,
    sleep: float,
    resume: bool,
):
    """번역 후편집 (Post-editing).
    
    Judge 모델을 사용하여 번역 결과를 개선합니다.
    """
    config: ConfigManager = ctx.obj["config"]
    
    from .corpus_cli import cmd_postedit
    import argparse
    
    args = argparse.Namespace(
        config=str(config.config_path),
        input=input_file,
        output=output,
        temperature=temperature,
        max_tokens=max_tokens,
        sleep=sleep,
        resume=resume,
        flush_every=20,
    )
    
    return cmd_postedit(args)


# =============================================================================
# ragas-eval 명령
# =============================================================================

@cli.command("ragas-eval")
@click.option("--input", "-i", "input_file", required=True, help="QA JSONL 입력 파일 경로")
@click.option("--output", "-o", required=True, help="RAGAS 결과 JSON 출력 경로")
@click.option("--limit", type=int, default=None, help="평가 샘플 수 제한")
@click.option("--llm-role", default="judge", help="LLM 역할 (generator|judge)")
@click.option("--llm-model", default=None, help="평가용 LLM 모델명 오버라이드")
@click.option("--llm-base-url", default=None, help="평가용 LLM base URL 오버라이드")
@click.option("--llm-api-key", default=None, help="평가용 LLM API 키 오버라이드")
@click.option("--emb-model", default=None, help="임베딩 모델명 (로컬 HuggingFace)")
@click.option("--emb-device", default=None, help="임베딩 디바이스 (cuda|cpu)")
@click.option("--metric", "metrics", multiple=True, help="RAGAS metric (복수 지정 가능)")
@click.option("--batch-size", type=int, default=None, help="RAGAS 배치 크기")
@click.option("--save-per-sample", is_flag=True, help="샘플별 점수 저장")
@click.option("--no-progress", is_flag=True, help="진행률 표시 비활성화")
@click.pass_context
def ragas_eval(
    ctx: click.Context,
    input_file: str,
    output: str,
    limit: Optional[int],
    llm_role: str,
    llm_model: Optional[str],
    llm_base_url: Optional[str],
    llm_api_key: Optional[str],
    emb_model: Optional[str],
    emb_device: Optional[str],
    metrics: tuple,
    batch_size: Optional[int],
    save_per_sample: bool,
    no_progress: bool,
):
    """QA 데이터셋에 대한 RAGAS 평가 실행."""
    from .validation.ragas_eval import load_qa_jsonl, run_ragas_eval

    config: ConfigManager = ctx.obj["config"]
    items = load_qa_jsonl(Path(input_file), limit=limit)
    summary, per_sample = run_ragas_eval(
        items,
        config=config,
        llm_role=llm_role,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        emb_model=emb_model,
        emb_device=emb_device,
        metric_names=metrics or None,
        batch_size=batch_size,
        show_progress=not no_progress,
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input": str(input_file),
        "num_samples": len(items),
        "metrics": list(metrics) if metrics else None,
        "summary": summary,
    }
    if save_per_sample:
        payload["per_sample"] = per_sample

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    click.echo(f"RAGAS 결과 저장: {output_path}")


# =============================================================================
# ragas-compare 명령
# =============================================================================

@cli.command("ragas-compare")
@click.option("--baseline", required=True, help="원본 QA JSONL 경로")
@click.option("--improved", required=True, help="개선본 QA JSONL 경로")
@click.option("--output", "-o", required=True, help="비교 결과 JSON 출력 경로")
@click.option("--limit", type=int, default=None, help="평가 샘플 수 제한")
@click.option("--llm-role", default="judge", help="LLM 역할 (generator|judge)")
@click.option("--llm-model", default=None, help="평가용 LLM 모델명 오버라이드")
@click.option("--llm-base-url", default=None, help="평가용 LLM base URL 오버라이드")
@click.option("--llm-api-key", default=None, help="평가용 LLM API 키 오버라이드")
@click.option("--emb-model", default=None, help="임베딩 모델명 (로컬 HuggingFace)")
@click.option("--emb-device", default=None, help="임베딩 디바이스 (cuda|cpu)")
@click.option("--metric", "metrics", multiple=True, help="RAGAS metric (복수 지정 가능)")
@click.option("--batch-size", type=int, default=None, help="RAGAS 배치 크기")
@click.option("--no-progress", is_flag=True, help="진행률 표시 비활성화")
@click.pass_context
def ragas_compare(
    ctx: click.Context,
    baseline: str,
    improved: str,
    output: str,
    limit: Optional[int],
    llm_role: str,
    llm_model: Optional[str],
    llm_base_url: Optional[str],
    llm_api_key: Optional[str],
    emb_model: Optional[str],
    emb_device: Optional[str],
    metrics: tuple,
    batch_size: Optional[int],
    no_progress: bool,
):
    """원본/개선본 QA 데이터셋의 RAGAS 결과 비교."""
    from .validation.ragas_eval import load_qa_jsonl, run_ragas_compare

    config: ConfigManager = ctx.obj["config"]
    baseline_items = load_qa_jsonl(Path(baseline))
    improved_items = load_qa_jsonl(Path(improved))

    baseline_map = {item.get(QAFields.ID): item for item in baseline_items}
    improved_map = {item.get(QAFields.ID): item for item in improved_items}
    common_ids = [item.get(QAFields.ID) for item in baseline_items if item.get(QAFields.ID) in improved_map]
    if limit:
        common_ids = common_ids[:limit]

    aligned_baseline = [baseline_map[i] for i in common_ids if i in baseline_map]
    aligned_improved = [improved_map[i] for i in common_ids if i in improved_map]

    result = run_ragas_compare(
        aligned_baseline,
        aligned_improved,
        config=config,
        llm_role=llm_role,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        emb_model=emb_model,
        emb_device=emb_device,
        metric_names=metrics or None,
        batch_size=batch_size,
        show_progress=not no_progress,
    )
    result.update(
        {
            "baseline": {"path": baseline, "num_samples": len(aligned_baseline), **result["baseline"]},
            "improved": {"path": improved, "num_samples": len(aligned_improved), **result["improved"]},
        }
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    click.echo(f"RAGAS 비교 결과 저장: {output_path}")


# =============================================================================
# config 명령 (설정 확인)
# =============================================================================

@cli.command("config")
@click.option("--show-all", is_flag=True, help="전체 설정 출력")
@click.option("--key", default=None, help="특정 설정 키 조회 (점 표기법)")
@click.pass_context
def show_config(ctx: click.Context, show_all: bool, key: Optional[str]):
    """현재 설정 확인.
    
    \b
    예시:
        python -m dataset_pipeline.cli config --show-all
        python -m dataset_pipeline.cli config --key llm.generator.model
    """
    config: ConfigManager = ctx.obj["config"]
    
    if key:
        value = config.get(key)
        if value is None:
            click.echo(f"키를 찾을 수 없습니다: {key}", err=True)
            raise SystemExit(1)
        click.echo(f"{key}: {value}")
    elif show_all:
        click.echo(config.debug_dump(mask_secrets=True))
    else:
        # 기본: 주요 설정만 출력
        click.echo("=== LLM 설정 ===")
        try:
            gen = config.get_llm_config("generator")
            click.echo(f"Generator: {gen.get('model')} @ {gen.get('base_url')}")
        except ValueError:
            click.echo("Generator: 미설정")
        
        try:
            judge = config.get_llm_config("judge")
            click.echo(f"Judge: {judge.get('model')} @ {judge.get('base_url')}")
        except ValueError:
            click.echo("Judge: 미설정")
        
        hf_token = config.get_huggingface_token()
        click.echo(f"\nHuggingFace Token: {'설정됨' if hf_token else '미설정'}")
        
        click.echo(f"\n설정 파일: {config.config_path}")
        click.echo(f"Secrets 파일: {config.secrets_path} ({'존재' if config.secrets_path.exists() else '미존재'})")


# =============================================================================
# 메인 엔트리포인트
# =============================================================================

def main():
    """CLI 메인 함수."""
    cli(obj={})


if __name__ == "__main__":
    main()
