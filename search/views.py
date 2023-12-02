from django.shortcuts import render, redirect
from django.contrib import messages
from .bsbi import BSBIIndex
import ir_datasets


def home(request):
    return render(request, 'index.html')


def search(request):
    query = request.GET.get('query')

    if query is None:
        return redirect('home')

    page = int(request.GET.get('page', 0))
    docstore = ir_datasets.load('antique').docs_store()

    result = []
    start = page * 10
    end = start + 10
    BSBI_instance = BSBIIndex()

    docs = list(BSBI_instance.retrieve_bm25(query, k=100))

    if len(docs) == 0:
        messages.error(request, f"Hasil {query} tidak ditemukan", extra_tags="danger")
        return redirect('home')

    if start > len(docs):
        start = 0
        end = 10

    for i, (score, doc) in enumerate(docs):
        if start <= i < end:
            content = docstore.get(doc).text
            result.append((doc, doc.split('_')[0], content))

    response = {
        'query': query,
        'result': result,
        'pages': range(((len(docs) - 1) // 10) + 1),
        'curr_page': page,
    }

    return render(request, 'search.html', response)


def detail(request, doc_id):
    docstore = ir_datasets.load('antique').docs_store()

    try:
        doc = docstore.get(doc_id)
        response = {
            'title': doc.doc_id.split('_')[0],
            'text': doc.text
        }

        return render(request, 'detail.html', response)
    except KeyError:
        messages.error(request, f"Dokumen {doc_id} tidak ditemukan", extra_tags="danger")
        return redirect('home')
